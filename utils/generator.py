import random
import string
from itertools import zip_longest

import intxeger

from . import engine


class GeneratorException(Exception):
    pass


class Generator:
    def __init__(self, max_repeat, max_n_strings, max_string_length):
        self.max_repeat = max_repeat
        self.max_n_strings = max_n_strings
        self.max_string_length = max_string_length

    def generate_positive_strings_from_union_regex(self, ast, is_test):
        max_n_strings = self.max_n_strings * (2 if is_test else 1)
        at_least = 3 if is_test else 2
        collected_groups = []
        for child in ast.children:
            sub_positive_strings = set()
            child_regex = child.get_regex()
            node = intxeger.build(child_regex, use_optimization=False, max_repeat=self.max_repeat)
            sampler = node.iterator()
            length = node.length
            for _ in range(length):
                candidate = next(sampler)
                if engine.fullmatch(child_regex, candidate) and len(candidate) <= self.max_string_length:
                    sub_positive_strings.add(candidate)
                if len(sub_positive_strings) >= max_n_strings:
                    break
            if len(sub_positive_strings) == 0:
                raise GeneratorException('Not enough positive strings generated.')
            collected_groups.append(list(sub_positive_strings))
        positive_strings = []
        seen = set()
        for batch in zip_longest(*collected_groups):
            for item in batch:
                if item is not None and item not in seen:
                    positive_strings.append(item)
                    seen.add(item)
                    if len(positive_strings) >= max_n_strings:
                        break
            if len(positive_strings) >= max_n_strings:
                break
        if len(positive_strings) < at_least:
            raise GeneratorException('Not enough positive strings generated.')
        positive_strings = positive_strings[::-1]
        if is_test:
            validation_positive_strings = positive_strings[: len(positive_strings) // 2]
            positive_strings = positive_strings[len(positive_strings) // 2 :]
        else:
            validation_positive_strings = []
        return positive_strings, validation_positive_strings

    def generate_positive_strings(self, ast, is_test):
        if ast.type == 'union':
            return self.generate_positive_strings_from_union_regex(ast, is_test)
        max_n_strings = self.max_n_strings * (2 if is_test else 1)
        at_least = 3 if is_test else 2
        regex = ast.get_regex()
        positive_strings = set()
        node = intxeger.build(regex, use_optimization=False, max_repeat=self.max_repeat)
        sampler = node.iterator()
        length = node.length
        for _ in range(length):
            candidate = next(sampler)
            if engine.fullmatch(regex, candidate) and len(candidate) <= self.max_string_length:
                positive_strings.add(candidate)
            if len(positive_strings) >= max_n_strings:
                break
        if len(positive_strings) < at_least:
            raise GeneratorException('Not enough positive strings generated.')
        positive_strings = list(positive_strings)
        if is_test:
            validation_positive_strings = positive_strings[: len(positive_strings) // 2]
            positive_strings = positive_strings[len(positive_strings) // 2 :]
        else:
            validation_positive_strings = []
        return positive_strings, validation_positive_strings

    def generate_negative_strings(self, ast, positive_strings, validation_positive_strings, is_test):
        regex = ast.get_regex()
        negative_strings = set()
        victims = positive_strings + validation_positive_strings
        for victim in victims:
            for i in range(len(victim)):
                candidate = victim[:i] + victim[i + 1 :]
                if not engine.fullmatch(regex, candidate) and len(candidate) <= self.max_string_length:
                    negative_strings.add(candidate)
            for i in range(len(victim) + 1):
                for c in string.printable:
                    candidate = victim[:i] + c + victim[i:]
                    if not engine.fullmatch(regex, candidate) and len(candidate) <= self.max_string_length:
                        negative_strings.add(candidate)
            for i in range(len(victim)):
                for c in string.printable:
                    if victim[i] == c:
                        continue
                    candidate = victim[:i] + c + victim[i + 1 :]
                    if not engine.fullmatch(regex, candidate) and len(candidate) <= self.max_string_length:
                        negative_strings.add(candidate)
        negative_strings = list(negative_strings)
        random.shuffle(negative_strings)
        negative_strings = negative_strings[: self.max_n_strings * (2 if is_test else 1)]
        if is_test:
            validation_negative_strings = negative_strings[: len(negative_strings) // 2]
            negative_strings = negative_strings[len(negative_strings) // 2 :]
        else:
            validation_negative_strings = []
        return negative_strings, validation_negative_strings

    def generate_strings(self, ast, is_test):
        positive_strings, validation_positive_strings = self.generate_positive_strings(ast, is_test)
        negative_strings, validation_negative_strings = self.generate_negative_strings(
            ast, positive_strings, validation_positive_strings, is_test
        )
        return (
            positive_strings,
            negative_strings,
            validation_positive_strings,
            validation_negative_strings,
        )
