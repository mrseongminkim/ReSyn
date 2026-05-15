import os
import pickle
import time

from datasets import load_dataset, load_from_disk
from prettytable import PrettyTable
from tqdm import tqdm

from baselines.gpt import GPTSynthesizer
from baselines.oss import GPTOssSynthesizer
from baselines.trivial import TrivialSynthesizer

from .engine import fullmatch
from .exceptions import SynthesisFailure
from .normalizer import normalize
from .splitters import OracleSplitter


def load_or_download_dataset(split, repo_id='mrseongminkim/ReSyn', cache_dir='data/datasets'):
    local_path = os.path.join(cache_dir, split)
    if os.path.exists(local_path):
        return load_from_disk(local_path)
    dataset = load_dataset(repo_id, split=split)
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(local_path)
    return dataset


class Evaluator:
    def __init__(self, synthesizer, logger, split_by='source', deduple_by_signature=True):
        self.synthesizer = synthesizer
        self.logger = logger
        self.benchmark = load_or_download_dataset('benchmark')
        self.split_by = split_by
        if deduple_by_signature:
            self.benchmark = self._deduplicate_by_signature(self.benchmark)
        self.trivial_synthesizer = TrivialSynthesizer()
        self._reset_stats()

    def evaluate_and_save_all(self, output_path):
        os.makedirs('logs/experiments_pickles', exist_ok=True)
        results = {}
        dataset = self.benchmark
        progress_bar = tqdm(dataset, ncols=160)
        for id, data in enumerate(progress_bar):
            positive_strings = data['positive_strings']
            valid_positive_strings = data['valid_positive_strings']
            negative_strings = data['negative_strings']
            valid_negative_strings = data['valid_negative_strings']
            ground_truth_regex = data['regex']
            regex = None
            synthesis_success = True
            start_time = time.time()
            try:
                if isinstance(self.synthesizer, OracleSplitter):
                    regex = self.synthesizer.synthesize(positive_strings, negative_strings, ground_truth_regex)
                elif isinstance(self.synthesizer, GPTSynthesizer) or isinstance(self.synthesizer, GPTOssSynthesizer):
                    regex = self.synthesizer.synthesize(positive_strings, negative_strings, id)
                else:
                    regex = self.synthesizer.synthesize(positive_strings, negative_strings)
            except SynthesisFailure:
                pass
            end_time = time.time()
            if not regex:
                synthesis_success = False
                regex = self.trivial_synthesizer.synthesize(positive_strings, negative_strings)
            mcc = self._calculate_mcc(regex, valid_positive_strings, valid_negative_strings)
            example_match = self._check_example_match(regex, valid_positive_strings, valid_negative_strings)
            results[id] = {
                'regex': regex,
                'source': data['source'],
                'depth': data['depth'],
                'synthesis_success': synthesis_success,
                'ground_truth_regex': ground_truth_regex,
                'positive_strings': positive_strings,
                'negative_strings': negative_strings,
                'valid_positive_strings': valid_positive_strings,
                'valid_negative_strings': valid_negative_strings,
                'synthesis_time': end_time - start_time,
                'mcc': mcc,
                'example_match': example_match,
            }
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

    def _deduplicate_by_signature(self, dataset):
        seen_signatures = set()
        unique_indices = []
        for idx, example in enumerate(dataset):
            ast = normalize(example['regex'])
            signature = ast.get_signature()
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_indices.append(idx)
        return dataset.select(unique_indices)

    def _get_result_table(self):
        table = PrettyTable()
        table.field_names = [
            'Dataset',
            'Synthesized',
            'Example_Match',
            'Under_Approx.',
            'Over_Approx.',
            'Both',
            'Time(s)',
            'Len(Synth.)',
            'Len(Ex.Mat.)',
            'Len Ratio',
            'MCC',
        ]
        return table

    def evaluate(self):
        result_table = self._get_result_table()
        if self.split_by == 'source':
            structured = self.benchmark.filter(lambda x: x['source'] == 'structured')
            snort = self.benchmark.filter(lambda x: x['source'] == 'snort')
            regexlib = self.benchmark.filter(lambda x: x['source'] == 'regexlib')
            targets = [('structured', structured), ('snort', snort), ('regexlib', regexlib)]
        else:
            unique_depths = sorted(set([x['depth'] for x in self.benchmark]))
            targets = [
                (f'depth_{depth}', self.benchmark.filter(lambda x, d=depth: x['depth'] == d)) for depth in unique_depths if depth < 6
            ]
            if any(depth >= 6 for depth in unique_depths):
                targets.append(('depth_6+', self.benchmark.filter(lambda x: x['depth'] >= 6)))
        for target, dataset in targets:
            result = self._evaluate(dataset)
            try:
                self.logger.info(
                    f'Sub Regex Fail: {self.synthesizer.sub_regex_fail}, Normalize Fail: {self.synthesizer.normalize_fail}, Accept Neg Fail: {self.synthesizer.accept_neg_fail}'
                )
            except AttributeError:
                pass
            result_table.add_row([target] + result)
        self.logger.info('Result Table\n' + result_table.get_string())

    def _calculate_mcc(self, regex, positive_strings, negative_strings):
        tp = sum(1 for s in positive_strings if fullmatch(regex, s) is not None)
        tn = sum(1 for s in negative_strings if fullmatch(regex, s) is None)
        fp = sum(1 for s in negative_strings if fullmatch(regex, s) is not None)
        fn = sum(1 for s in positive_strings if fullmatch(regex, s) is None)
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _check_example_match(self, regex, positive_strings, negative_strings):
        under_approximated = any(fullmatch(regex, s) is None for s in positive_strings)
        over_approximated = any(fullmatch(regex, s) is not None for s in negative_strings)
        if under_approximated and over_approximated:
            self.both += 1
        elif under_approximated:
            self.under_approximated += 1
        elif over_approximated:
            self.over_approximated += 1
        return not (under_approximated or over_approximated)

    def _reset_stats(self):
        self.synthesized = 0
        self.example_match = 0
        self.under_approximated = 0
        self.over_approximated = 0
        self.both = 0
        self.time_to_synthesize = 0.0
        self.avg_length_synthesized = 0.0
        self.avg_length_example_match = 0.0
        self.total_synth_length = 0.0
        self.total_gt_length = 0.0
        self.total_mcc = 0.0

    def _evaluate(self, dataset):
        self._reset_stats()
        dataset = dataset.filter(lambda x: not x['is_substring'])
        total = len(dataset)
        progress_bar = tqdm(dataset, ncols=160)
        for data in progress_bar:
            positive_strings = data['positive_strings']
            valid_positive_strings = data['valid_positive_strings']
            negative_strings = data['negative_strings']
            valid_negative_strings = data['valid_negative_strings']
            ground_truth_regex = data['regex']
            id = data['id']
            regex = None
            start_time = time.time()
            try:
                if isinstance(self.synthesizer, OracleSplitter):
                    regex = self.synthesizer.synthesize(positive_strings, negative_strings, ground_truth_regex)
                elif isinstance(self.synthesizer, GPTSynthesizer) or isinstance(self.synthesizer, GPTOssSynthesizer):
                    regex = self.synthesizer.synthesize(positive_strings, negative_strings, id)
                else:
                    regex = self.synthesizer.synthesize(positive_strings, negative_strings)
            except SynthesisFailure:
                pass
            end_time = time.time()
            if regex:
                self.time_to_synthesize += end_time - start_time
                self.synthesized += 1
                self.avg_length_synthesized += len(regex)
                if self._check_example_match(regex, valid_positive_strings, valid_negative_strings):
                    self.example_match += 1
                    self.avg_length_example_match += len(regex)
            else:
                regex = self.trivial_synthesizer.synthesize(positive_strings, negative_strings)
            self.total_synth_length += len(regex)
            self.total_gt_length += len(ground_truth_regex)
            mcc = self._calculate_mcc(regex, valid_positive_strings, valid_negative_strings)
            self.total_mcc += mcc
            progress_bar.set_postfix(Synthesized=f'{self.synthesized}/{total}', Example_Match=f'{self.example_match}/{total}')
        self.logger.info(f'{self.synthesized} / {total} synthesized, {self.example_match} / {total} example match')
        avg_mcc = self.total_mcc / total
        return [
            f'{self.synthesized / total * 100:.2f}%\n{self.synthesized}/{total}',
            f'{self.example_match / total * 100:.2f}%\n{self.example_match}/{total}',
            f'{self.under_approximated / (self.synthesized + 1e-8) * 100:.2f}%\n{self.under_approximated}/{self.synthesized}',
            f'{self.over_approximated / (self.synthesized + 1e-8) * 100:.2f}%\n{self.over_approximated}/{self.synthesized}',
            f'{self.both / (self.synthesized + 1e-8) * 100:.2f}%\n{self.both}/{self.synthesized}',
            f'{self.time_to_synthesize / (self.synthesized + 1e-8):.2f}s',
            f'{(self.avg_length_synthesized / (self.synthesized + 1e-8)):.2f}',
            f'{(self.avg_length_example_match / (self.example_match + 1e-8)):.2f}',
            f'{(self.total_synth_length / (self.total_gt_length + 1e-8)):.2f}',
            f'{avg_mcc:.4f}',
        ]
