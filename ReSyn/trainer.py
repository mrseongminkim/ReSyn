import argparse
import os
from datetime import datetime
from time import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from config import CheckPointConfig, RouterConfig

from .dataset import RegexVocabulary, SegmenterVocabulary, get_dataloader
from .model import Partitioner, Router, Segmenter, Set2Regex


def load_or_download_dataset(split, repo_id='mrseongminkim/ReSyn', cache_dir='data/datasets'):
    local_path = os.path.join(cache_dir, split)
    if os.path.exists(local_path):
        return load_from_disk(local_path)
    dataset = load_dataset(repo_id, split=split)
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(local_path)
    return dataset


class Trainer:
    def __init__(self, model, optimizer, wandb_logger, model_name, mode, model_dir, epochs=999, patience=10, max_grad_norm=1.0):
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.wandb_logger = wandb_logger
        self.model_name = model_name
        self.mode = mode
        self.model_dir = model_dir
        self.epochs = epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.current_epoch = 0
        self.counter = 0
        self.best_accuracy = -float('inf')
        self.best_valid_loss = float('inf')
        self.terminate = False
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join('checkpoints', model_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def log_metrics(self, metrics: dict):
        if self.wandb_logger is None:
            return
        self.wandb_logger.log_metrics(metrics, step=self.current_epoch)

    def early_stop(self, valid_metric, is_accuracy=True):
        self.counter += 1
        if is_accuracy:
            if self.best_accuracy < valid_metric:
                self.best_accuracy = valid_metric
                self.counter = 0
                torch.save(self.model.state_dict(), f'checkpoints/{self.model_dir}/{self.model_name}_best_acc.pt')
            elif self.counter >= self.patience:
                self.terminate = True
        else:
            if self.best_valid_loss > valid_metric:
                self.best_valid_loss = valid_metric
                self.counter = 0
                torch.save(self.model.state_dict(), f'checkpoints/{self.model_dir}/{self.model_name}_best_loss.pt')
            elif self.counter >= self.patience:
                self.terminate = True

    def load_loaders(self, is_test=False, target='snort'):
        if is_test:
            dataset = load_or_download_dataset(target)
            self.valid_loader = get_dataloader(dataset, shuffle=False, mode=self.mode)
        else:
            train_dataset = load_or_download_dataset('train')
            valid_dataset = load_or_download_dataset('valid')
            self.train_loader = get_dataloader(train_dataset, shuffle=True, mode=self.mode)
            self.valid_loader = get_dataloader(valid_dataset, shuffle=False, mode=self.mode)
            iter_per_batch = len(self.train_loader)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-8, total_iters=1_000)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iter_per_batch * self.epochs, eta_min=1e-6)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[1_000]
            )

    def run(self):
        self.load_loaders()
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            train_start_time = time()
            train_loss = self.train()
            train_time = time() - train_start_time
            valid_start_time = time()
            valid_loss = self.validate()
            valid_time = time() - valid_start_time
            self.log_metrics(
                {
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_time_sec': train_time,
                    'valid_time_sec': valid_time,
                }
            )
            if self.terminate:
                self.log_metrics({'best_valid_loss': self.best_valid_loss})
                break

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def train(self):
        pass

    def validate(self):
        pass


class PartitionerTrainer(Trainer):
    def __init__(self, epochs=999, patience=10, max_grad_norm=1.0, lr=0.0005, is_test=False):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_name = now if not is_test else CheckPointConfig.partitioner.split('_best')[0]
        self.mode = 'partitioner'
        self.model_dir = 'partitioner'
        self.model: Partitioner = Partitioner().cuda()
        self.criterion = torch.nn.NLLLoss().cuda()
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        self.wandb_logger = WandbLogger(
            project='ReSyn',
            name='Partitioner',
            log_model=False,
        )
        super().__init__(
            self.model,
            self.optimizer,
            self.wandb_logger,
            self.model_name,
            self.mode,
            self.model_dir,
            epochs,
            patience,
            max_grad_norm,
        )

    def train(self):
        losses = 0
        self.model.train()
        for data in tqdm(self.train_loader, ncols=80):
            pos = data['pos'].cuda()
            labels = data['labels'].cuda()
            decoder_input = labels[:, :-1]
            labels = labels[:, 1:].reshape(-1)
            log_probs = self.model(pos, decoder_input)
            log_probs = log_probs.view(-1, log_probs.size(-1))
            loss: torch.Tensor = self.criterion(log_probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            losses += loss.item()
        return losses / len(self.train_loader)

    def validate(self, is_test=False):
        losses = 0
        total_characters = 0
        total_strings = 0
        total_correct_characters = 0
        total_correct_strings = 0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.valid_loader, ncols=80):
                pos = data['pos'].cuda()  # batch_size, n_strings, max_len
                labels = data['labels'].cuda()  # batch_size, n_strings
                decoder_inputs = labels[:, :-1]  # batch_size, n_strings - 1
                decoder_outputs = labels[:, 1:].reshape(-1)
                log_probs = self.model(pos, decoder_inputs)  # batch_size, n_strings - 1, vocab_size
                predictions = log_probs.argmax(dim=-1)  # batch_size, n_strings - 1
                log_probs = log_probs.view(-1, log_probs.size(-1))
                loss = self.criterion(log_probs, decoder_outputs)
                n_correct_characters, n_correct_strings = self.calculate_accuracy(predictions, labels[:, 1:])
                total_correct_characters += n_correct_characters
                total_correct_strings += n_correct_strings
                total_characters += labels.size(0) * labels.size(1)
                total_strings += labels.size(0)
                losses += loss.item()
        if total_characters > 0:
            self.log_metrics(
                {
                    'val_char_acc': total_correct_characters / total_characters,
                    'val_str_acc': total_correct_strings / total_strings,
                }
            )
        if len(self.valid_loader):
            losses /= len(self.valid_loader)
        else:
            losses = float('inf')
        if not is_test:
            self.early_stop(losses, is_accuracy=False)
        return losses

    def calculate_accuracy(self, predictions, labels):
        correct_predictions = predictions == labels
        n_correct_characters = correct_predictions.sum().item()
        correct_strings = correct_predictions.all(dim=-1)
        n_correct_strings = correct_strings.sum().item()
        return n_correct_characters, n_correct_strings


class RouterTrainer(Trainer):
    def __init__(self, epochs=999, patience=10, max_grad_norm=1.0, lr=0.0005, is_test=False):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_name = now if not is_test else CheckPointConfig.router.split('_best')[0]
        self.mode = 'router'
        self.model_dir = 'router'
        self.model: Router = Router().cuda()
        counts = torch.tensor([RouterConfig.n_concat, RouterConfig.n_union, RouterConfig.n_no_op], dtype=torch.float)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        self.criterion = torch.nn.NLLLoss(weight=weights).cuda()
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        self.wandb_logger = WandbLogger(
            project='ReSyn',
            name='Router',
            log_model=False,
        )
        super().__init__(
            self.model,
            self.optimizer,
            self.wandb_logger,
            self.model_name,
            self.mode,
            self.model_dir,
            epochs,
            patience,
            max_grad_norm,
        )
        self.wandb_logger.log_hyperparams(
            {
                'class_weight_concat': weights[0].item(),
                'class_weight_union': weights[1].item(),
                'class_weight_no_op': weights[2].item(),
            }
        )

    def train(self):
        losses = 0
        self.model.train()
        for data in tqdm(self.train_loader, ncols=80):
            pos = data['pos'].cuda()
            operators = data['operators'].cuda()
            log_probs = self.model(pos)
            loss: torch.Tensor = self.criterion(log_probs, operators)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            losses += loss.item()
        return losses / len(self.train_loader)

    def validate(self, is_test=False):
        losses = 0
        all_predictions = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.valid_loader, ncols=80):
                pos = data['pos'].cuda()
                operators = data['operators'].cuda()  # batch_size
                log_probs = self.model(pos)  # batch_size, 3
                loss = self.criterion(log_probs, operators)
                losses += loss.item()
                predictions = log_probs.argmax(dim=-1)  # batch_size
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(operators.cpu().numpy())
        class_names = ['concat', 'union', 'no-op']
        macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        report = classification_report(all_labels, all_predictions, labels=[0, 1, 2], target_names=class_names, zero_division=0)
        cm = confusion_matrix(all_labels, all_predictions)
        self.log_metrics({'val_macro_f1': macro_f1})
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.summary['classification_report'] = report
            self.wandb_logger.experiment.summary['confusion_matrix'] = cm.tolist()
        losses /= len(self.valid_loader)
        if not is_test:
            self.early_stop(losses, is_accuracy=False)
        return losses


class Set2RegexTrainer(Trainer):
    def __init__(self, epochs=999, patience=10, max_grad_norm=1.0, lr=0.0005, is_test=False):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_name = now if not is_test else CheckPointConfig.set2regex.split('_best')[0]
        self.model_dir = 'set2regex'
        self.mode = 'set2regex'
        self.model: Set2Regex = Set2Regex().cuda()
        self.criterion = torch.nn.NLLLoss(ignore_index=RegexVocabulary.pad_token_index).cuda()
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        self.wandb_logger = WandbLogger(
            project='ReSyn',
            name='Set2Regex',
            log_model=False,
        )
        super().__init__(
            self.model,
            self.optimizer,
            self.wandb_logger,
            self.model_name,
            self.mode,
            self.model_dir,
            epochs,
            patience,
            max_grad_norm,
        )

    def train(self):
        losses = 0
        self.model.train()
        for data in tqdm(self.train_loader, ncols=80):
            strings = data['strings'].cuda()
            types = data['types'].cuda()
            regex = data['regex'].cuda()
            decoder_inputs = regex[:, :-1]
            labels = regex[:, 1:].reshape(-1)
            log_probs = self.model(strings, types, decoder_inputs)
            log_probs = log_probs.view(-1, log_probs.size(-1))
            loss: torch.Tensor = self.criterion(log_probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            losses += loss.item()
        return losses / len(self.train_loader)

    def validate(self, is_test=False):
        losses = 0
        total_characters = 0
        total_strings = 0
        total_correct_characters = 0
        total_correct_strings = 0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.valid_loader, ncols=80):
                strings = data['strings'].cuda()
                types = data['types'].cuda()
                regex = data['regex'].cuda()
                decoder_inputs = regex[:, :-1]
                labels = regex[:, 1:]
                log_probs = self.model(strings, types, decoder_inputs)
                predictions = log_probs.argmax(dim=-1)  # batch_size, max_regex_len - 1
                log_probs = log_probs.view(-1, log_probs.size(-1))
                loss = self.criterion(log_probs, regex[:, 1:].reshape(-1))
                n_characters, n_correct_characters, n_correct_strings = self.calculate_accuracy(predictions, labels)
                total_characters += n_characters
                total_correct_characters += n_correct_characters
                total_correct_strings += n_correct_strings
                total_strings += regex.size(0)
                losses += loss.item()
        self.log_metrics(
            {
                'val_char_acc': total_correct_characters / total_characters,
                'val_str_acc': total_correct_strings / total_strings,
            }
        )
        losses /= len(self.valid_loader)
        if not is_test:
            self.early_stop(losses, is_accuracy=False)
        return losses

    def calculate_accuracy(self, predictions, labels):
        pad = labels == RegexVocabulary.pad_token_index
        n_characters = (~pad).sum().item()
        n_correct_characters = ((predictions == labels) & ~pad).sum().item()
        correct_predictions = (predictions == labels) | pad
        correct_strings = correct_predictions.all(dim=-1)
        n_correct_strings = correct_strings.sum().item()
        return n_characters, n_correct_characters, n_correct_strings


class SegmenterLightning(pl.LightningModule):
    def __init__(self, lr=0.0005, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = Segmenter()
        self.dec_vocab = self.model.dec_vocab
        self.criterion = nn.NLLLoss(ignore_index=SegmenterVocabulary.pad_token_index)
        self.validation_step_outputs = []

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids, decoder_input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        decoder_input_ids = labels[:, :-1]
        decoder_labels = labels[:, 1:]
        logits = self(input_ids, decoder_input_ids)
        loss = self.criterion(logits.view(-1, self.dec_vocab), decoder_labels.reshape(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        decoder_input_ids = labels[:, :-1]
        decoder_labels = labels[:, 1:]
        logits = self(input_ids, decoder_input_ids)
        loss = self.criterion(logits.view(-1, self.dec_vocab), decoder_labels.reshape(-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        predictions = logits.argmax(dim=-1)
        n_char, n_corr_char, n_corr_str = self.calculate_accuracy(predictions, decoder_labels)
        metrics = {
            'n_char': n_char,
            'n_corr_char': n_corr_char,
            'n_str': labels.size(0),
            'n_corr_str': n_corr_str,
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        total_n_char = sum(x['n_char'] for x in self.validation_step_outputs)
        total_n_corr_char = sum(x['n_corr_char'] for x in self.validation_step_outputs)
        total_n_str = sum(x['n_str'] for x in self.validation_step_outputs)
        total_n_corr_str = sum(x['n_corr_str'] for x in self.validation_step_outputs)
        metrics_tensor = torch.tensor(
            [total_n_char, total_n_corr_char, total_n_str, total_n_corr_str], device=self.device, dtype=torch.float
        )
        gathered_metrics = self.all_gather(metrics_tensor)
        global_sums = gathered_metrics.sum(dim=0)
        global_n_char = global_sums[0]
        global_n_corr_char = global_sums[1]
        global_n_str = global_sums[2]
        global_n_corr_str = global_sums[3]
        char_acc = global_n_corr_char / global_n_char if global_n_char > 0 else 0.0
        str_acc = global_n_corr_str / global_n_str if global_n_str > 0 else 0.0
        self.log('val_char_acc', char_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_str_acc', str_acc, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def calculate_accuracy(self, predictions, labels):
        pad = labels == SegmenterVocabulary.pad_token_index
        n_characters = (~pad).sum().item()
        n_correct_characters = ((predictions == labels) & ~pad).sum().item()
        correct_predictions = (predictions == labels) | pad
        correct_strings = correct_predictions.all(dim=-1)
        n_correct_strings = correct_strings.sum().item()
        return n_characters, n_correct_characters, n_correct_strings

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        if self.trainer.train_dataloader:
            steps_per_epoch = len(self.trainer.train_dataloader) // self.trainer.accumulate_grad_batches
        else:
            steps_per_epoch = 1000
        total_steps = steps_per_epoch * self.hparams.max_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, total_iters=1000)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[1000])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }


def train_segmenter():
    train_dataset = load_or_download_dataset('train')
    valid_dataset = load_or_download_dataset('valid')
    train_loader = get_dataloader(train_dataset, shuffle=True, batch_size=64, num_proc=16, mode='segmenter')
    val_loader = get_dataloader(valid_dataset, shuffle=False, batch_size=64, num_proc=16, mode='segmenter')
    model = SegmenterLightning(lr=0.0005, max_epochs=100)
    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints/segmenter', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/segmenter',
        filename=f'{timestamp}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy='ddp',
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=WandbLogger(
            project='ReSyn',
            name='Segmenter',
            log_model=False,
        ),
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save best checkpoint in standard format
    if checkpoint_callback.best_model_path:
        import shutil

        best_ckpt_path = checkpoint_callback.best_model_path
        best_loss_path = os.path.join('checkpoints/segmenter', f'{timestamp}_best_loss.pt')
        shutil.copy(best_ckpt_path, best_loss_path)
        print(f'\nBest checkpoint saved to: {best_loss_path}')
        print(f'Best val_loss: {checkpoint_callback.best_model_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ReSyn models.')
    parser.add_argument(
        '--model', type=str, required=True, choices=['partitioner', 'router', 'set2regex', 'segmenter'], help='Model to train'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (ignored for segmenter, which uses all 4 GPUs via DDP)')
    args = parser.parse_args()
    if args.model == 'segmenter':
        train_segmenter()
    else:
        torch.cuda.set_device(args.gpu)
        if args.model == 'partitioner':
            trainer = PartitionerTrainer()
            trainer.run()
        elif args.model == 'router':
            trainer = RouterTrainer()
            trainer.run()
        elif args.model == 'set2regex':
            trainer = Set2RegexTrainer()
            trainer.run()
