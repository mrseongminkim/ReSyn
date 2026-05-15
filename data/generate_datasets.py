import csv
import json
import os
import random
import warnings
from collections import defaultdict
from concurrent.futures import TimeoutError, as_completed
from pathlib import Path

import jsonlines
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk
from pebble import ProcessPool
from tqdm.auto import tqdm

from baselines.FOREST.forest.parse_examples import parse_file
from utils.generator import GeneratorException
from utils.normalizer import NormalizeException, normalize
from utils.workers import generate_worker, init_worker, substrings_task


def clean_automatark_regexes() -> None:
    """Cleaning https://github.com/lorisdanto/automatark/tree/master/regex"""
    with open('data/raw/regexlib-clean.re', 'r') as f:
        regexes = sorted(set(f.readlines()))
    with open('data/clean/regexlib.txt', 'w') as f:
        for regex in regexes:
            f.write(regex)
    with open('data/raw/snort-clean.re', 'r') as f:
        regexes = sorted(set(f.readlines()))
    with open('data/clean/snort.txt', 'w') as f:
        for regex in regexes:
            regex = eval(regex)
            regex = regex[1 : regex.rindex('/')]
            f.write(repr(regex) + '\n')


def clean_python_regexes() -> None:
    """Cleaning https://github.com/softwarekitty/tour_de_source/blob/master/analysis/pattern_tracking/corpusPatterns.txt"""
    with open('data/raw/corpusPatterns.txt', 'r') as f:
        regexes = sorted(set(f.readlines()))
    with open('data/clean/python.txt', 'w') as f:
        for regex in regexes:
            f.write(regex)


def clean_polyglot_regexes() -> None:
    """Cleaning https://github.com/SBULeeLab/LinguaFranca-FSE19/blob/master/data/production-regexes/uniq-regexes-8.json"""
    with jsonlines.open('data/raw/uniq-regexes-8.json') as reader:
        regexes = list(set(obj['pattern'] for obj in reader))
    regexes = sorted([regex for regex in regexes if isinstance(regex, str)])
    with open('data/clean/polyglot.txt', 'w') as f:
        for regex in regexes:
            f.write(repr(regex) + '\n')


def clean_splitregex_regexes() -> None:
    """Cleaning regexes from SplitRegex"""
    with open('data/raw/snort_jalc_regex.txt') as f:
        splitregex_snort = sorted(set(f.readlines()))
    with open('data/raw/regexlib_jalc_regex.txt') as f:
        splitregex_regexlib = sorted(set(f.readlines()))
    with open('data/clean/splitregex_snort.txt', 'w') as f:
        for regex in splitregex_snort:
            f.write(regex)
    with open('data/clean/splitregex_regexlib.txt', 'w') as f:
        for regex in splitregex_regexlib:
            f.write(regex)


def clean_prax_regexes() -> None:
    """Cleaning https://github.com/saujasv/generating-pragmatic-examples/tree/main/data"""
    literal_train = set()
    with open('data/raw/listener-train-specs-suffix-idx.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            literal_train.add(row[-1])
    literal_valid = set()
    with open('data/raw/listener-validation-specs-suffix-idx.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            literal_valid.add(row[-1])
    with open('data/raw/annotation-pool-sub30.txt', 'r') as f:
        annotation = set(f.readlines())
    with open('data/raw/heldout-programs-sub30.txt', 'r') as f:
        heldout = set(f.readlines())
    with open('data/raw/pragmatic-target-programs.txt', 'r') as f:
        pragmatic = set(f.readlines())
    hft_train = set()
    with open('data/raw/verified_split_train_set.json', 'r') as f:
        hft_data = json.load(f)
        for data in hft_data:
            hft_train.add(data[0])
    hft_valid = []
    with open('data/raw/verified_data_listener_loss_s=full.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        tsv_data = [row[-1] for row in reader]
    hft_valid = set(tsv_data)
    structured = sorted(
        literal_train.union(literal_valid).union(annotation).union(heldout).union(pragmatic).union(hft_train).union(hft_valid)
    )
    with open('data/clean/structured_train.txt', 'w') as f:
        for program in structured:
            f.write(repr(program) + '\n')
    replay_regexes = set()
    with open('data/raw/replay-record.json', 'r') as f:
        replay = json.load(f)
        for record in replay.values():
            regex = record['program']
            replay_regexes.add(regex)
    with open('data/clean/structured_test.txt', 'w') as f:
        for program in sorted(replay_regexes):
            f.write(repr(program) + '\n')


def clean_forest_regexes() -> None:
    """Cleaning https://github.com/mrseongminkim/FOREST/tree/master/benchmarks"""
    raw_path = Path('baselines/FOREST/benchmarks')
    forest_regexes = set()
    for file in map(str, sorted(raw_path.glob('*.txt'))):
        _, _, _, ground_truth = parse_file(file)
        ground_truth = ground_truth.split(',')
        regex = ','.join(filter(lambda s: not s.lstrip().startswith('$'), ground_truth))
        forest_regexes.add(regex)
    regexes = sorted(forest_regexes)
    with open('data/clean/forest.txt', 'w') as f:
        for regex in regexes:
            f.write(repr(regex) + '\n')


def clean_regexes() -> None:
    os.makedirs('data/clean', exist_ok=True)
    clean_automatark_regexes()
    clean_python_regexes()
    clean_polyglot_regexes()
    clean_splitregex_regexes()
    clean_prax_regexes()
    clean_forest_regexes()


def normalize_regexes() -> None:
    targets = (
        'splitregex_regexlib',
        'splitregex_snort',
        'forest',
        'regexlib',
        'snort',
        'structured_test',
        'python',
        'structured_train',
        'polyglot',
    )

    def read_file(target: str) -> list[str]:
        with open(f'data/clean/{target}.txt') as f:
            regexes = f.readlines()
        return list(map(eval, regexes))

    def normalize_regexes(target: list[str], is_anonymized: bool = False, is_splitregex: bool = False) -> list[str]:
        normalized = set()
        for regex in tqdm(target):
            try:
                ast = normalize(regex, is_anonymized, is_splitregex)
                regex1 = ast.get_regex()
                regex2 = normalize(regex1).get_regex()
                assert regex1 == regex2, f'Normalization is not idempotent\n{repr(regex)}\n{repr(regex1)}\n{repr(regex2)}'
                normalized.add(regex2)
            except NormalizeException:
                pass
        return list(sorted(normalized))

    def save_normalized(target: str) -> None:
        regexes = read_file(target)
        is_splitregex = target.startswith('splitregex')
        normalized = normalize_regexes(regexes, is_splitregex=is_splitregex)
        with open(f'data/normalized/{target}.txt', 'w') as f:
            for regex in normalized:
                f.write(repr(regex) + '\n')

    os.makedirs('data/normalized', exist_ok=True)
    for target in targets:
        save_normalized(target)


def generate_benchmark():
    def read_file(target):
        with open(f'data/normalized/{target}.txt') as f:
            regexes = f.readlines()
        return list(map(eval, regexes))

    def filter_regexes(ast_list):
        filtered = []
        for ast in ast_list:
            if ast.type == 'union' and len(ast.children) > MAX_N_STRINGS:
                continue
            if ast.type == 'concat' and len(ast.children) > MAX_STRING_LENGTH:
                continue
            if len(ast.get_regex()) > MAX_REGEX_LENGTH:
                continue
            filtered.append(ast)
        return filtered

    def normalize_regexes(target):
        target = read_file(target)
        normalized = []
        for regex in tqdm(target):
            ast = normalize(regex)
            normalized.append(ast)
        filtered = filter_regexes(normalized)
        return filtered

    def generate_dataset(ast_list):
        dataset = []
        with ProcessPool(initializer=init_worker) as pool:
            futures = {pool.schedule(generate_worker, args=(ast, True), timeout=60): ast for ast in ast_list}
            for future in tqdm(as_completed(futures), total=len(ast_list)):
                try:
                    result = future.result()
                    dataset.append(result)
                except (GeneratorException, TimeoutError):
                    continue
        dataset = Dataset.from_list(dataset)
        return dataset

    def add_depth(item):
        regex = item['regex']
        ast = normalize(regex)
        item['depth'] = ast.get_depth()
        return item

    structured = generate_dataset(normalize_regexes('structured_test'))
    snort = generate_dataset(normalize_regexes('snort'))
    regexlib = generate_dataset(normalize_regexes('regexlib'))
    structured = structured.add_column('source', ['structured'] * len(structured))
    snort = snort.add_column('source', ['snort'] * len(snort))
    regexlib = regexlib.add_column('source', ['regexlib'] * len(regexlib))
    benchmark = concatenate_datasets([structured, snort, regexlib])
    benchmark = benchmark.map(add_depth)
    benchmark.save_to_disk('data/datasets/benchmark')


def generate_train_datasets():
    features = Features(
        {
            'regex': Value('string'),
            'positive_strings': Sequence(Value('string')),
            'negative_strings': Sequence(Value('string')),
            'valid_positive_strings': Sequence(Value('string')),
            'valid_negative_strings': Sequence(Value('string')),
            'is_substring': Value('bool'),
        }
    )

    def read_file(target):
        with open(f'data/normalized/{target}.txt') as f:
            regexes = f.readlines()
        return list(map(eval, regexes))

    def get_ast_list():
        regexes = set()
        for target in ('polyglot', 'python', 'structured_train'):
            target_regexes = read_file(target)
            regexes.update(target_regexes)
        ast_list = []
        for regex in tqdm(regexes):
            ast = normalize(regex)
            ast_list.append(ast)
        return ast_list

    def get_sub_ast_list(ast_list):
        sub_ast_list = []
        expanded_regexes = set(ast.get_regex() for ast in ast_list)
        expandable = [ast for ast in ast_list if ast.type in {'concat', 'union', 'repetition'}]
        while True:
            target = []
            for ast in tqdm(expandable):
                match ast.type:
                    case 'concat' | 'union':
                        for i in range(len(ast.children)):
                            child = ast.children[i]
                            child_regex = child.get_regex()
                            if child_regex not in expanded_regexes:
                                expanded_regexes.add(child_regex)
                                target.append(ast.children[i])
                    case 'repetition':
                        child = ast.child
                        child_regex = child.get_regex()
                        if child_regex not in expanded_regexes:
                            expanded_regexes.add(child_regex)
                            target.append(ast.child)
            sub_ast_list.extend(target)
            expandable = [ast for ast in target if ast.type in {'concat', 'union', 'repetition'}]
            if not expandable:
                break
        return sub_ast_list

    def get_benchmark_signature():
        benchmark = load_from_disk('data/datasets/benchmark')
        benchmark_signature = set()
        for item in benchmark:
            regex = item['regex']
            ast = normalize(regex)
            benchmark_signature.add(ast.get_signature())
        return benchmark_signature

    def filter_regexes(ast_list):
        benchmark_signature = get_benchmark_signature()
        filtered = []
        for ast in tqdm(ast_list):
            if ast.type == 'union' and len(ast.children) > MAX_N_STRINGS:
                continue
            if ast.type == 'concat' and len(ast.children) > MAX_STRING_LENGTH:
                continue
            if len(ast.get_regex()) > MAX_REGEX_LENGTH:
                continue
            if ast.get_signature() in benchmark_signature:
                continue
            filtered.append(ast)
        return filtered

    def generate_strings(ast_list):
        dataset = []
        with ProcessPool(initializer=init_worker) as pool:
            futures = {pool.schedule(generate_worker, args=(ast, False), timeout=60): ast for ast in ast_list}
            for future in tqdm(as_completed(futures), total=len(ast_list)):
                try:
                    result = future.result()
                    dataset.append(result)
                except (GeneratorException, TimeoutError):
                    continue
        dataset = Dataset.from_list(dataset, features=features)
        return dataset

    def get_substring_dataset(strings):
        substrings_dataset = []
        with ProcessPool(initializer=init_worker) as pool:
            futures = {pool.schedule(substrings_task, args=(item,), timeout=60): item for item in strings}
            for future in tqdm(as_completed(futures), total=len(strings)):
                try:
                    result = future.result()
                    substrings_dataset.extend(result)
                except TimeoutError:
                    continue
        substrings_dataset = Dataset.from_list(substrings_dataset, features=features)
        return substrings_dataset

    def filter_fn(item, benchmark_signature):
        ast = normalize(item['regex'])
        if ast.type == 'union' and len(ast.children) > MAX_N_STRINGS:
            return False
        if ast.type == 'concat' and len(ast.children) > MAX_STRING_LENGTH:
            return False
        if len(ast.get_regex()) > MAX_REGEX_LENGTH:
            return False
        if ast.get_signature() in benchmark_signature:
            return False
        return True

    ast_list = get_ast_list()
    sub_ast_list = get_sub_ast_list(ast_list)
    final_ast_list = ast_list + sub_ast_list
    filtered_ast_list = filter_regexes(final_ast_list)
    strings = generate_strings(filtered_ast_list)
    benchmark_signature = get_benchmark_signature()
    substrings = get_substring_dataset(strings).filter(filter_fn, fn_kwargs={'benchmark_signature': benchmark_signature})
    train = concatenate_datasets([strings, substrings])

    typed_items = defaultdict(list)
    for item in tqdm(train):
        regex = item['regex']
        ast = normalize(regex)
        typed_items[ast.type].append(item)

    train_items = []
    valid_items = []
    train_signatures = set()

    for _, items in typed_items.items():
        random.shuffle(items)
        split_index = int(len(items) * 0.9)
        type_train_items = items[:split_index]
        type_valid_items = items[split_index:]

        for item in tqdm(type_train_items):
            ast = normalize(item['regex'])
            train_signatures.add(ast.get_signature())
        train_items.extend(type_train_items)

        for item in tqdm(type_valid_items):
            ast = normalize(item['regex'])
            if ast.get_signature() not in train_signatures:
                valid_items.append(item)

    train_dataset = Dataset.from_list(train_items)
    valid_dataset = Dataset.from_list(valid_items)
    train_dataset.save_to_disk('data/datasets/train')
    valid_dataset.save_to_disk('data/datasets/valid')


if __name__ == '__main__':
    random.seed(42)
    warnings.filterwarnings('ignore', category=FutureWarning)

    MAX_N_STRINGS = 10
    MAX_STRING_LENGTH = 110
    MAX_REGEX_LENGTH = 110

    clean_regexes()
    normalize_regexes()
    generate_benchmark()
    generate_train_datasets()
