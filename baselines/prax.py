import os
import subprocess

import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils.engine import EngineException, fullmatch
from utils.exceptions import SynthesisFailure, get_smallest_named_character_class
from utils.normalizer import NormalizeException, normalize


class Prax:
    def __init__(self, do_sample=False, debug=False):
        model_path = 'checkpoints/prax/final'
        if not os.path.exists(model_path):
            snapshot_download(repo_id='mrseongminkim/ReSyn-byt5-small', local_dir=model_path)
        gen_config = {
            'do_sample': do_sample,
            'max_new_tokens': 128,
            'num_return_sequences': 500 if do_sample else 1,
            'top_p': 0.9 if do_sample else 1.0,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'decoder_start_token_id': 0,
        }
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
        self.gen_config = GenerationConfig(**gen_config)
        self.device = device
        self.program_special_token = '<extra_id_124>'
        self.utterances_special_token = '<extra_id_123>'
        self.utterances_to_string = lambda spec: ''.join([f'<extra_id_{i}>{s}{label}' for i, (s, label) in enumerate(spec)])
        self.debug = debug
        self.fallback = False

    def decode(self, sequences):
        return [[''.join([chr(i - 3) for i in sequence if 3 <= i <= 258]) for sequence in beam] for beam in sequences]

    def _synthesize(self, specs):
        batch_size = 1
        specs_string = [self.utterances_to_string(spec) for spec in specs]
        specs_tokens = self.tokenizer([f'{self.utterances_special_token}{c}' for c in specs_string], return_tensors='pt', padding=True).to(
            self.device
        )
        decoder_inputs = self.tokenizer([self.program_special_token for _ in specs], return_tensors='pt', add_special_tokens=False).to(
            self.device
        )
        outputs = self.model.generate(
            input_ids=specs_tokens.input_ids,
            attention_mask=specs_tokens.attention_mask,
            decoder_input_ids=decoder_inputs.input_ids,
            generation_config=self.gen_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
        regexes = self.decode(
            outputs.sequences.reshape((batch_size, self.gen_config.num_return_sequences, outputs.sequences.shape[-1])).tolist()
        )[0]
        log_probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
        gen_probs = torch.gather(log_probs, 2, outputs.sequences[:, 2:].unsqueeze(-1)).squeeze(-1)
        gen_probs.masked_fill_(gen_probs.isinf(), 0)
        scores = gen_probs.sum(-1).reshape(batch_size, -1).tolist()[0]
        specs = specs[0]
        candidates = sorted(zip(regexes, scores), key=lambda x: x[1], reverse=True)
        if self.debug:
            _candidates = [regex for regex, _ in candidates]
            print(set(_candidates))
        for regex, _ in candidates:
            consistent = True
            try:
                ast = normalize(regex)
                regex = ast.get_regex()
                needs_question_mark = False
                for string, label in specs:
                    if label == '-':
                        break
                    if not string and not fullmatch(regex, string):
                        needs_question_mark = True
                        break
                if needs_question_mark:
                    regex = f'({regex})?'
            except NormalizeException:
                continue
            try:
                for string, label in specs:
                    if label == '+' and not fullmatch(regex, string):
                        consistent = False
                        break
                    if label == '-' and fullmatch(regex, string):
                        consistent = False
                        break
            except EngineException:
                consistent = False
            if consistent:
                return regex
        raise SynthesisFailure

    def synthesize(self, positive_strings: list[str], negative_strings: list[str]) -> str:
        specs = []
        for s in positive_strings:
            specs.append((s, '+'))
        for s in negative_strings:
            specs.append((s, '-'))
        try:
            return self._synthesize([specs])
        except SynthesisFailure:
            if self.fallback:
                return get_smallest_named_character_class(positive_strings, negative_strings)
            else:
                raise


class DataCollatorForSeq2Seq:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.padding = True
        self.max_length = None
        self.pad_to_multiple_of = None
        self.label_pad_token_id = -100
        self.return_tensors = 'pt'

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature['labels'] for feature in features]
        max_label_length = max(len(label) for label in labels)
        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature['labels']))
            decoder_inputs_remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature['decoder_input_ids']))
            feature['labels'] = feature['labels'] + remainder if padding_side == 'right' else remainder + feature['labels']
            feature['decoder_input_ids'] = (
                feature['decoder_input_ids'] + decoder_inputs_remainder
                if padding_side == 'right'
                else decoder_inputs_remainder + feature['decoder_input_ids']
            )
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        return features


def linearize_strings(examples):
    contexts = []
    for positive_strings, negative_strings in zip(examples['positive_strings'], examples['negative_strings']):
        context = '<extra_id_123>'
        for i, positive_string in enumerate(positive_strings):
            context += f'<extra_id_{i}>{positive_string}+'
        for i, negative_string in enumerate(negative_strings, start=len(positive_strings)):
            context += f'<extra_id_{i}>{negative_string}-'
        contexts.append(context)
    return {'context': contexts}


def preprocess_function(examples):
    model_inputs = tokenizer(examples['context'], text_target=examples['regex'])  # input_ids, attention_mask, labels
    bos = tokenizer.convert_tokens_to_ids('<extra_id_124>')  # 383 as int
    decoder_input_ids = [[bos, *inp[:-1]] for inp in model_inputs['labels']]  # labels shifted right with bos at start
    return {**model_inputs, 'decoder_input_ids': decoder_input_ids}


def load_or_download_dataset(split, repo_id='mrseongminkim/ReSyn', cache_dir='data/datasets'):
    local_path = os.path.join(cache_dir, split)
    if os.path.exists(local_path):
        return load_from_disk(local_path)
    dataset = load_dataset(repo_id, split=split)
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(local_path)
    return dataset


def train_prax(warmup_dataset=False):
    global tokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained('google/byt5-small', cache_dir='checkpoints/prax/cache')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small', cache_dir='checkpoints/prax/cache')

    train_dataset = load_or_download_dataset('train')
    valid_dataset = load_or_download_dataset('valid')

    train_dataset = train_dataset.map(linearize_strings, batched=True, batch_size=1000, num_proc=16).select_columns(['context', 'regex'])
    valid_dataset = valid_dataset.map(linearize_strings, batched=True, batch_size=1000, num_proc=16).select_columns(['context', 'regex'])

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['regex', 'context'], batch_size=1000, num_proc=16)
    valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=['regex', 'context'], batch_size=1000, num_proc=16)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    if warmup_dataset:
        return

    training_args = Seq2SeqTrainingArguments(
        run_name='Prax',
        output_dir='checkpoints/prax/runs',
        seed=42,
        warmup_ratio=0.03,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        bf16=True,
        tf32=True,
        dataloader_num_workers=8,
        group_by_length=True,
        eval_strategy='steps',
        eval_steps=2000,
        save_strategy='steps',
        save_steps=2000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        logging_steps=100,
        report_to='wandb',
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    trainer.save_model('checkpoints/prax/final')


if __name__ == '__main__':
    os.environ['WANDB_PROJECT'] = 'ReSyn'

    if os.environ.get('LOCAL_RANK') is None and os.environ.get('RANK') is None and os.environ.get('WORLD_SIZE') is None:
        train_prax(warmup_dataset=True)
        module_name = getattr(__spec__, 'name', None)
        if module_name:
            cmd = ['torchrun', '--nproc_per_node=4', '--module', module_name]
        else:
            script_path = os.path.abspath(__file__)
            cmd = ['torchrun', '--nproc_per_node=4', script_path]
        subprocess.run(cmd, check=True)
    else:
        train_prax()
