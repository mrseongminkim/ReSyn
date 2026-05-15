from collections import defaultdict
from itertools import islice, product

from baselines.forest import forest_split
from ReSyn.server import PartitionerServer, RouterServer, SegmenterServer
from utils.engine import escape, fullmatch
from utils.exceptions import SynthesisFailure
from utils.normalizer import NormalizeException, normalize


def synthesize_from_single_string(positive_string: str, negative_strings: list[str]):
    if positive_string == '':
        empty_string_token = escape('\x01')
        return f'({empty_string_token})?'  # this effectively matches only the empty string since \x01 is not in the alphabet
    return escape(positive_string)


class ForestSplitter:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def synthesize(self, positive_strings, negative_strings):
        regex = self._synthesize(positive_strings, negative_strings)
        try:
            ast = normalize(regex)
            regex = ast.get_regex()
            if '' in positive_strings and not fullmatch(regex, ''):
                regex = f'({regex})?'
        except NormalizeException:
            raise SynthesisFailure
        if any(fullmatch(regex, neg) for neg in negative_strings):
            raise SynthesisFailure
        if any(not fullmatch(regex, pos) for pos in positive_strings):
            breakpoint()
        return regex

    def _synthesize(self, positive_strings, negative_strings):
        if len(positive_strings) == 0:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        elif len(positive_strings) == 1:
            return synthesize_from_single_string(positive_strings[0], negative_strings)
        sub_problems = forest_split(positive_strings)
        if len(sub_problems) == 1:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        sub_regexes = []
        for sub_positive_strings in sub_problems:
            if len(sub_positive_strings) == 0:
                breakpoint()
            elif len(sub_positive_strings) == 1:
                if sub_positive_strings[0] == '':
                    continue
                else:
                    sub_regex = synthesize_from_single_string(sub_positive_strings[0], [])
            else:
                sub_regex = self.synthesizer.synthesize(sub_positive_strings, [])
            sub_regexes.append(f'({sub_regex})')
        if not sub_regexes:
            breakpoint()  # Debug
        return ''.join(sub_regexes)


class ReSynSplitter:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.router = RouterServer()
        self.partitioner = PartitionerServer()
        self.segmenter = SegmenterServer()
        # statistics
        self.sub_regex_fail = 0
        self.normalize_fail = 0
        self.accept_neg_fail = 0

    def synthesize(self, positive_strings, negative_strings):
        try:
            regex = self._synthesize(positive_strings, negative_strings)
        except SynthesisFailure:
            self.sub_regex_fail += 1
            raise
        try:
            ast = normalize(regex)
            regex = ast.get_regex()
            if '' in positive_strings and not fullmatch(regex, ''):
                regex = f'({regex})?'
        except NormalizeException:
            self.normalize_fail += 1
            raise SynthesisFailure
        if any(fullmatch(regex, neg) for neg in negative_strings):
            self.accept_neg_fail += 1
            raise SynthesisFailure
        if any(not fullmatch(regex, pos) for pos in positive_strings):
            breakpoint()
        return regex

    def _synthesize(self, positive_strings, negative_strings, previous_ast_type=None):
        if len(positive_strings) == 0:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        elif len(positive_strings) == 1:
            return synthesize_from_single_string(positive_strings[0], negative_strings)
        ast_type = self.router.classify(positive_strings)
        match ast_type:
            case 'concat':
                if previous_ast_type == 'concat':
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_strings_list = self.segmenter.split(positive_strings)
                if len(sub_strings_list) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    breakpoint()  # this will never happen
                sub_regexes = []
                for sub_positive_strings in sub_strings_list:
                    if sub_positive_strings == ['']:
                        continue
                    sub_regex = self._synthesize(sub_positive_strings, [], 'concat')
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not sub_regexes:
                    breakpoint()  # Debug
                return ''.join(sub_regexes)
            case 'union':
                if previous_ast_type == 'union':
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_strings_list = self.partitioner.split(positive_strings)
                if len(sub_strings_list) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_regexes = []
                for sub_positive_strings in sub_strings_list:
                    sub_regex = self._synthesize(sub_positive_strings, [], 'union')
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not sub_regexes:
                    breakpoint()  # Debug
                return '|'.join(sub_regexes)
            case _:
                return self.synthesizer.synthesize(positive_strings, negative_strings)


class OracleSplitter:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def split_positive_strings(self, positive_strings: list[str], ast):
        subregexes = []
        for i, child in enumerate(ast.children):
            subregexes.append(f'(?P<G{i}>{child.get_regex()})')
        named_regex = '|'.join(subregexes) if ast.type == 'union' else ''.join(subregexes)
        substrings_set = defaultdict(set)
        for positive_string in positive_strings:
            match = fullmatch(named_regex, positive_string)
            if match:
                groups = match.groupdict()
                for key, val in groups.items():
                    if val is not None:
                        child_idx = int(key[1:])
                        substrings_set[child_idx].add(val)
        return [(ast.children[i].get_regex(), list(substrings_set[i])) for i in range(len(ast.children)) if substrings_set[i]]

    def _synthesize(self, positive_strings, negative_strings, gt_regex):
        if len(positive_strings) == 0:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        elif len(positive_strings) == 1:
            return synthesize_from_single_string(positive_strings[0], negative_strings)
        ast = normalize(gt_regex)
        match ast.type:
            case 'concat' | 'union':
                substrings_set = self.split_positive_strings(positive_strings, ast)
                if len(substrings_set) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                elif all(len(positive_substrings) == 1 for _, positive_substrings in substrings_set):
                    if ast.type == 'concat':
                        breakpoint()  # this will never happen
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_regexes = []
                for gt_sub_regex, positive_substrings in substrings_set:
                    sub_regex = self._synthesize(positive_substrings, [], gt_sub_regex)
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not len(sub_regexes):
                    breakpoint()  # Debug
                sub_regexes = '|'.join(sub_regexes) if ast.type == 'union' else ''.join(sub_regexes)
                return sub_regexes
            case _:
                return self.synthesizer.synthesize(positive_strings, negative_strings)

    def synthesize(self, positive_strings, negative_strings, gt_regex):
        regex = self._synthesize(positive_strings, negative_strings, gt_regex)
        try:
            ast = normalize(regex)
            regex = ast.get_regex()
            if '' in positive_strings and not fullmatch(regex, ''):
                regex = f'({regex})?'
        except NormalizeException:
            raise SynthesisFailure
        if any(fullmatch(regex, neg) for neg in negative_strings):
            raise SynthesisFailure
        if any(not fullmatch(regex, pos) for pos in positive_strings):
            breakpoint()
        return regex


class FallbackSplitter:
    def __init__(self, synthesizer, max_candidates_per_subproblem=100_000, max_total_combinations=1_000_000):
        self.synthesizer = synthesizer
        self.router = RouterServer()
        self.partitioner = PartitionerServer()
        self.segmenter = SegmenterServer()
        self.max_try = 1_000
        self.max_candidates_per_subproblem = max_candidates_per_subproblem
        self.max_total_combinations = max_total_combinations

    def synthesize(self, positive_strings, negative_strings):
        count = 0
        for regex in self._synthesize_all(positive_strings, negative_strings):
            try:
                ast = normalize(regex)
                regex = ast.get_regex()
                if '' in positive_strings and not fullmatch(regex, ''):
                    regex = f'({regex})?'
            except NormalizeException:
                continue
            if any(fullmatch(regex, neg) for neg in negative_strings):
                count += 1
                if count >= self.max_try:
                    break
                else:
                    continue
            if any(not fullmatch(regex, pos) for pos in positive_strings):
                breakpoint()
            return regex
        return self.synthesizer.synthesize(positive_strings, negative_strings)  # final fallback for non-decomposable cases

    def _synthesize_all(self, positive_strings, negative_strings, previous_ast_type=None):
        if len(positive_strings) == 0:
            yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)
            return
        elif len(positive_strings) == 1:
            yield synthesize_from_single_string(positive_strings[0], negative_strings)
            return
        ast_type = self.router.classify(positive_strings)
        match ast_type:
            case 'concat':
                if previous_ast_type == 'concat':
                    yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)
                    return
                sub_strings_list = self.segmenter.split(positive_strings)
                if len(sub_strings_list) == 1:
                    yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)
                    return
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    breakpoint()  # this will never happen
                sub_regex_generators = []
                for sub_positive_strings in sub_strings_list:
                    if sub_positive_strings == ['']:
                        continue
                    # Limit candidates per sub-problem to prevent memory explosion
                    sub_gen = self._synthesize_all(sub_positive_strings, [], 'concat')
                    sub_regex_generators.append(list(islice(sub_gen, self.max_candidates_per_subproblem)))
                # Limit total combinations to prevent memory explosion
                for sub_regexes in islice(product(*sub_regex_generators), self.max_total_combinations):
                    yield ''.join(f'({sub_regex})' for sub_regex in sub_regexes)
            case 'union':
                if previous_ast_type == 'union':
                    yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)
                    return
                sub_strings_list = self.partitioner.split(positive_strings)
                if len(sub_strings_list) == 1:
                    yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)
                    return
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)
                    return
                sub_regex_generators = []
                for sub_positive_strings in sub_strings_list:
                    # Limit candidates per sub-problem to prevent memory explosion
                    sub_gen = self._synthesize_all(sub_positive_strings, [], 'union')
                    sub_regex_generators.append(list(islice(sub_gen, self.max_candidates_per_subproblem)))
                # Limit total combinations to prevent memory explosion
                for sub_regexes in islice(product(*sub_regex_generators), self.max_total_combinations):
                    yield '|'.join(f'({sub_regex})' for sub_regex in sub_regexes)
            case _:
                yield from self.synthesizer.synthesize_all(positive_strings, negative_strings)


class FixedRecursiveSplitter:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.partitioner = PartitionerServer()
        self.segmenter = SegmenterServer()
        # statistics
        self.sub_regex_fail = 0
        self.normalize_fail = 0
        self.accept_neg_fail = 0

    def synthesize(self, positive_strings, negative_strings):
        try:
            regex = self._synthesize(positive_strings, negative_strings)
        except SynthesisFailure:
            self.sub_regex_fail += 1
            raise
        try:
            ast = normalize(regex)
            regex = ast.get_regex()
            if '' in positive_strings and not fullmatch(regex, ''):
                regex = f'({regex})?'
        except NormalizeException:
            self.normalize_fail += 1
            raise SynthesisFailure
        if any(fullmatch(regex, neg) for neg in negative_strings):
            self.accept_neg_fail += 1
            raise SynthesisFailure
        if any(not fullmatch(regex, pos) for pos in positive_strings):
            breakpoint()
        return regex

    def _synthesize(self, positive_strings, negative_strings, previous_ast_type='union'):
        if len(positive_strings) == 0:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        elif len(positive_strings) == 1:
            return synthesize_from_single_string(positive_strings[0], negative_strings)
        # ast_type = self.router.classify(positive_strings)
        ast_type = 'concat' if previous_ast_type == 'union' else 'union'
        match ast_type:
            case 'concat':
                if previous_ast_type == 'concat':
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_strings_list = self.segmenter.split(positive_strings)
                if len(sub_strings_list) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    breakpoint()  # this will never happen
                sub_regexes = []
                for sub_positive_strings in sub_strings_list:
                    if sub_positive_strings == ['']:
                        continue
                    sub_regex = self._synthesize(sub_positive_strings, [], 'concat')
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not sub_regexes:
                    breakpoint()  # Debug
                return ''.join(sub_regexes)
            case 'union':
                if previous_ast_type == 'union':
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_strings_list = self.partitioner.split(positive_strings)
                if len(sub_strings_list) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_regexes = []
                for sub_positive_strings in sub_strings_list:
                    sub_regex = self._synthesize(sub_positive_strings, [], 'union')
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not sub_regexes:
                    breakpoint()  # Debug
                return '|'.join(sub_regexes)
            case _:
                return self.synthesizer.synthesize(positive_strings, negative_strings)


class SegmenterOnlySplitter:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.segmenter = SegmenterServer()

    def synthesize(self, positive_strings, negative_strings):
        regex = self._synthesize(positive_strings, negative_strings)
        try:
            ast = normalize(regex)
            regex = ast.get_regex()
            if '' in positive_strings and not fullmatch(regex, ''):
                regex = f'({regex})?'
        except NormalizeException:
            raise SynthesisFailure
        if any(fullmatch(regex, neg) for neg in negative_strings):
            raise SynthesisFailure
        if any(not fullmatch(regex, pos) for pos in positive_strings):
            breakpoint()
        return regex

    def _synthesize(self, positive_strings, negative_strings):
        if len(positive_strings) == 0:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        elif len(positive_strings) == 1:
            return synthesize_from_single_string(positive_strings[0], negative_strings)
        # ast_type = self.router.classify(positive_strings)
        ast_type = 'concat'
        match ast_type:
            case 'concat':
                sub_strings_list = self.segmenter.split(positive_strings)
                if len(sub_strings_list) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_regexes = []
                for sub_positive_strings in sub_strings_list:
                    if sub_positive_strings == ['']:
                        continue
                    elif len(sub_positive_strings) == 1:
                        sub_regex = synthesize_from_single_string(sub_positive_strings[0], [])
                    else:
                        sub_regex = self.synthesizer.synthesize(sub_positive_strings, [])
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not sub_regexes:
                    breakpoint()  # Debug
                return ''.join(sub_regexes)
            case _:
                return self.synthesizer.synthesize(positive_strings, negative_strings)


class PartitionerOnlySplitter:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.partitioner = PartitionerServer()

    def synthesize(self, positive_strings, negative_strings):
        regex = self._synthesize(positive_strings, negative_strings)
        try:
            ast = normalize(regex)
            regex = ast.get_regex()
            if '' in positive_strings and not fullmatch(regex, ''):
                regex = f'({regex})?'
        except NormalizeException:
            raise SynthesisFailure
        if any(fullmatch(regex, neg) for neg in negative_strings):
            raise SynthesisFailure
        if any(not fullmatch(regex, pos) for pos in positive_strings):
            breakpoint()
        return regex

    def _synthesize(self, positive_strings, negative_strings):
        if len(positive_strings) == 0:
            return self.synthesizer.synthesize(positive_strings, negative_strings)
        elif len(positive_strings) == 1:
            return synthesize_from_single_string(positive_strings[0], negative_strings)
        # ast_type = self.router.classify(positive_strings)
        ast_type = 'union'
        match ast_type:
            case 'union':
                sub_strings_list = self.partitioner.split(positive_strings)
                if len(sub_strings_list) == 1:
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                elif all(len(sub_positive_strings) == 1 for sub_positive_strings in sub_strings_list):
                    return self.synthesizer.synthesize(positive_strings, negative_strings)
                sub_regexes = []
                for sub_positive_strings in sub_strings_list:
                    if len(sub_positive_strings) == 1:
                        sub_regex = synthesize_from_single_string(sub_positive_strings[0], [])
                    else:
                        sub_regex = self.synthesizer.synthesize(sub_positive_strings, [])
                    if sub_regex == '':
                        breakpoint()  # Debug
                    sub_regexes.append(f'({sub_regex})')
                if not sub_regexes:
                    breakpoint()  # Debug
                return '|'.join(sub_regexes)
            case _:
                return self.synthesizer.synthesize(positive_strings, negative_strings)
