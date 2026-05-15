from collections import defaultdict

from . import engine
from .generator import Generator
from .normalizer import normalize

MAX_REPEAT = 20
MAX_N_STRINGS = 10
MAX_STRING_LENGTH = 110


def init_worker():
    global generator
    generator = Generator(MAX_REPEAT, MAX_N_STRINGS, MAX_STRING_LENGTH)


def generate_worker(ast, is_test):
    pos, neg, val_pos, val_neg = generator.generate_strings(ast, is_test)
    return {
        'regex': ast.get_regex(),
        'positive_strings': pos,
        'negative_strings': neg,
        'valid_positive_strings': val_pos,
        'valid_negative_strings': val_neg,
        'is_substring': False,
    }


def substrings_task(data):
    results = []
    regex = data['regex']
    ast = normalize(regex)
    if ast.type not in ('union', 'concat'):
        return results
    positive_strings = data['positive_strings']
    valid_positive_strings = data['valid_positive_strings']
    subregexes = []
    for i, child in enumerate(ast.children):
        subregexes.append(f'(?P<G{i}>{child.get_regex()})')
    named_regex = '|'.join(subregexes) if ast.type == 'union' else ''.join(subregexes)
    sub_positive = defaultdict(set)
    sub_valid_positive = defaultdict(set)
    for string in positive_strings:
        match = engine.fullmatch(named_regex, string)
        if match:
            groups = match.groupdict()
            for key, val in groups.items():
                if val is not None:
                    child_idx = int(key[1:])
                    sub_positive[child_idx].add(val)
    for string in valid_positive_strings:
        match = engine.fullmatch(named_regex, string)
        if match:
            groups = match.groupdict()
            for key, val in groups.items():
                if val is not None:
                    child_idx = int(key[1:])
                    sub_valid_positive[child_idx].add(val)
    for i in range(len(ast.children)):
        if len(sub_positive[i]) < 2:
            continue
        results.append(
            {
                'regex': ast.children[i].get_regex(),
                'positive_strings': list(sub_positive[i]),
                'negative_strings': [],
                'valid_positive_strings': list(sub_valid_positive[i]),
                'valid_negative_strings': [],
                'is_substring': True,
            }
        )
    return results
