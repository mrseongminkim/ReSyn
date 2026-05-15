import random

from .builder import Builder, BuilderException
from .optimizer import Optimizer, OptimizerException
from .structure import Node


class NormalizeException(Exception):
    pass


builder = Builder()
optimizer = Optimizer()

counter = 0
anonymization_map = {}
anonymization_token = [chr(i) for i in list(range(3, 9)) + list(range(14, 32)) + [127]]  # leave 0, 1 2 for pad, empty string, empty set


def _check_assertion(node):
    global counter, anonymization_map, anonymization_token
    match node.type:
        case 'empty':
            raise AssertionError('Empty node after optimization')
        case 'literal':
            if len(node.literal) == 0:
                raise AssertionError('Literal with empty string after optimization')
            elif len(node.literal) == 1:
                pass
            elif node.literal in anonymization_map:
                node.literal = anonymization_map[node.literal]
            elif counter >= len(anonymization_token):
                raise NormalizeException('Too many unique literals to anonymize')
            else:
                new_literal = anonymization_token[counter]
                anonymization_map[node.literal] = new_literal
                node.literal = new_literal
                counter += 1
            assert len(node.literal) == 1, (
                f'Literal length greater than 1 after anonymization: {repr(node.literal)}, {repr(node.get_regex())}'
            )
        case 'characterclass':
            assert len(node.characters) > 1
        case 'repetition':
            _check_assertion(node.child)
        case 'concat' | 'union':
            for child in node.children:
                _check_assertion(child)


def check_assertion(node):
    original_regex = node.get_regex()
    while True:
        _check_assertion(node)
        current_regex = node.get_regex()
        while True:
            optimized_node = _normalize(current_regex)
            optimized_regex = optimized_node.get_regex()
            if optimized_regex == current_regex:
                break
            current_regex = optimized_regex
        if current_regex == original_regex:
            break
        node = _normalize(current_regex)
        original_regex = current_regex
    return node


def _normalize(regex: str, is_anonymized: bool = True):
    try:
        built_node = builder.build(regex, is_anonymized)
        regex = optimizer.optimize(built_node, valid_empty_token=True)
        return regex
    except (BuilderException, OptimizerException):
        raise NormalizeException


def normalize(regex: str, is_anonymized: bool = True, is_splitregex: bool = False) -> Node:
    global counter, anonymization_map, anonymization_token
    if is_splitregex:
        if '\x00' in regex:
            regex = regex.replace('\x00', '\n')
        if '\x01' in regex:
            regex = regex.replace('\x01', '\r')
        if '\x02' in regex:
            regex = regex.replace('\x02', '\t')
    if is_anonymized:
        ast = _normalize(regex)
    else:
        regex = _normalize(regex, is_anonymized).get_regex()
        while True:
            ast = _normalize(regex)
            anonymized_regex = ast.get_regex()
            if anonymized_regex == regex:
                random.shuffle(anonymization_token)
                counter = 0
                anonymization_map = {}
                try:
                    ast = check_assertion(ast)
                except AssertionError:
                    raise NormalizeException
                break
            regex = anonymized_regex
    return ast
