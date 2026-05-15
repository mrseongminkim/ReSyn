import string
from collections import defaultdict
from random import shuffle

import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from config import StringConfig
from utils.engine import escape, fullmatch
from utils.normalizer import normalize


class BucketBatchSampler(Sampler):
    def __init__(self, batch_size, indices_and_lengths, drop_last, shuffle):
        self.batch_size = batch_size
        self.indices_and_lengths = indices_and_lengths
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_map = self._generate_batch_map()
        self.batch_list = self._get_batch_list()
        self.length = len(self.batch_list)

    def _generate_batch_map(self):
        batch_map = defaultdict(list)
        for index, length in self.indices_and_lengths:
            batch_map[length].append(index)
        return batch_map

    def _shuffle_batch_map(self):
        for key in self.batch_map.keys():
            shuffle(self.batch_map[key])

    def _get_batch_list(self):
        batch_list = []
        self._shuffle_batch_map()
        for length, indices in self.batch_map.items():
            for batch in [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]:
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                else:
                    batch_list.append(batch)
        shuffle(batch_list)
        return batch_list

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in self.batch_list:
            yield i
        if self.shuffle:
            self.batch_list = self._get_batch_list()


class Vocabulary:
    @classmethod
    def convert_tokens_to_indices(cls, tokens):
        return [cls.token_to_index[token] for token in tokens]

    @classmethod
    def convert_indices_to_tokens(cls, indices):
        return [cls.index_to_token[index] for index in indices]

    @classmethod
    def get_vocabulary_size(cls):
        return len(cls.index_to_token)


class StringVocabulary(Vocabulary):
    special_tokens = ['<pad>', '<empty_string>', '<empty_set>']
    printable_tokens = list(string.printable)
    lcs_tokens = [chr(i) for i in list(range(3, 9)) + list(range(14, 32)) + [127]]
    index_to_token = special_tokens + printable_tokens + lcs_tokens
    token_to_index = {token: index for index, token in enumerate(index_to_token)}
    pad_token = '<pad>'
    empty_string_token = '<empty_string>'
    empty_set_token = '<empty_set>'
    pad_token_index = token_to_index[pad_token]
    empty_set_token_index = token_to_index[empty_set_token]


class PartitionerVocabulary(Vocabulary):
    labels = [chr(i) for i in range(StringConfig.max_n_strings)]
    index_to_token = labels
    token_to_index = {label: index for index, label in enumerate(index_to_token)}
    sos_token = '\x00'
    sos_token_index = token_to_index[sos_token]


class SegmenterVocabulary(Vocabulary):
    labels = [chr(i) for i in range(StringConfig.max_string_length)]
    special_tokens = ['<empty_string>', '<sos>', '<pad>']
    index_to_token = labels + special_tokens
    token_to_index = {label: index for index, label in enumerate(index_to_token)}
    pad_token = '<pad>'
    sos_token = '<sos>'
    empty_string_token = '<empty_string>'
    pad_token_index = token_to_index[pad_token]
    sos_token_index = token_to_index[sos_token]
    empty_string_token_index = token_to_index[empty_string_token]


class RouterVocabulary(Vocabulary):
    labels = ['concat', 'union', 'no-op']
    index_to_token = labels
    token_to_index = {label: index for index, label in enumerate(index_to_token)}


class RegexVocabulary(Vocabulary):
    special_tokens = ['<sos>', '<eos>', '<pad>']
    ascii_tokens = list(string.printable)
    escaped_tokens = [escape(chr(i)) for i in range(128)] + [r'\d', r'\D', r'\w', r'\W', r'\s', r'\S']
    tokens = list(sorted(set(ascii_tokens + escaped_tokens)))
    index_to_token = special_tokens + tokens
    token_to_index = {token: index for index, token in enumerate(index_to_token)}
    sos_token = '<sos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    sos_token_index = token_to_index[sos_token]
    pad_token_index = token_to_index[pad_token]
    eos_token_index = token_to_index[eos_token]


def add_strings_indices(example, target):
    strings_indices = []
    empty_string_token = StringVocabulary.empty_string_token
    empty_set_index = StringVocabulary.empty_set_token_index
    for s in example[target]:
        tokens = list(s) if s else [empty_string_token]
        strings_indices.append(StringVocabulary.convert_tokens_to_indices(tokens))
    return strings_indices if strings_indices else [[empty_set_index]]


def add_partitioner_labels(example):
    labels = example['labels']
    return PartitionerVocabulary.convert_tokens_to_indices(labels)


def add_segmenter_labels(example):
    labels_indices = []
    empty_string_token = SegmenterVocabulary.empty_string_token
    for label in example['labels']:
        tokens = list(label) if label else [empty_string_token]
        labels_indices.append(SegmenterVocabulary.convert_tokens_to_indices(tokens))
    return labels_indices


def add_router_labels(example):
    regex_type = example['regex_type']
    regex_type = 'no-op' if regex_type not in ('concat', 'union') else regex_type
    return RouterVocabulary.token_to_index[regex_type]


def add_regex_indices(example):
    i = 0
    tokens = []
    regex = example['regex']
    regex_length = len(regex)
    while i < regex_length:
        if regex[i] == '\\':
            tokens.append(regex[i : i + 2])
            i += 2
        else:
            tokens.append(regex[i])
            i += 1
    tokens = [RegexVocabulary.sos_token] + tokens + [RegexVocabulary.eos_token]
    regex_indices = RegexVocabulary.convert_tokens_to_indices(tokens)
    return regex_indices


def add_partitioner_indices(example, index):
    example['positive_strings_indices'] = add_strings_indices(example, target='positive_strings')
    example['labels_indices'] = add_partitioner_labels(example)
    p_size = len(example['positive_strings_indices'])
    example['indices_and_lengths'] = (index, p_size)
    return example


def add_segmenter_indices(example, index):
    example['positive_strings_indices'] = add_strings_indices(example, target='positive_strings')
    example['labels_indices'] = add_segmenter_labels(example)
    return example


def add_router_indices(example, index):
    example['positive_strings_indices'] = add_strings_indices(example, target='positive_strings')
    example['operator_indices'] = add_router_labels(example)
    p_size = len(example['positive_strings_indices'])
    example['indices_and_lengths'] = (index, p_size)
    return example


def add_set2regex_indices(example, index):
    example['positive_strings_indices'] = add_strings_indices(example, target='positive_strings')
    example['negative_strings_indices'] = add_strings_indices(example, target='negative_strings')
    example['regex_indices'] = add_regex_indices(example)
    p_size = len(example['positive_strings_indices'])
    n_size = len(example['negative_strings_indices'])
    example['indices_and_lengths'] = (index, p_size + n_size)
    return example


def partitioner_collate_fn(examples):
    batch_size = len(examples)
    n_strings = len(examples[0]['positive_strings_indices'])
    pos = []
    labels = []
    for example in examples:
        pos.extend(example['positive_strings_indices'])
        labels.append(example['labels_indices'])
    pos = pad_sequence([torch.tensor(s) for s in pos], batch_first=True, padding_value=StringVocabulary.pad_token_index).view(
        batch_size, n_strings, -1
    )
    labels = torch.tensor(labels)
    return {'pos': pos, 'labels': labels}


def segmenter_collate_fn(examples):
    string_separater = StringVocabulary.empty_set_token_index
    label_separater = SegmenterVocabulary.sos_token_index
    batch_size = len(examples)
    input_ids = []
    labels = []
    for example in examples:
        flattened = []
        for s in example['positive_strings_indices']:
            if flattened:
                flattened.append(string_separater)
            flattened.extend(s)
        input_ids.append(flattened)
        flattened = []
        for label in example['labels_indices']:
            flattened.extend([label_separater] + label)
        labels.append(flattened)
    input_ids = pad_sequence([torch.tensor(s) for s in input_ids], batch_first=True, padding_value=StringVocabulary.pad_token_index).view(
        batch_size, -1
    )
    labels = pad_sequence([torch.tensor(s) for s in labels], batch_first=True, padding_value=SegmenterVocabulary.pad_token_index).view(
        batch_size, -1
    )
    return {'input_ids': input_ids, 'labels': labels}


def router_collate_fn(examples):
    batch_size = len(examples)
    n_strings = len(examples[0]['positive_strings_indices'])
    pos = []
    operators = []
    for example in examples:
        pos.extend(example['positive_strings_indices'])
        operators.append(example['operator_indices'])
    pos = pad_sequence([torch.tensor(s) for s in pos], batch_first=True, padding_value=StringVocabulary.pad_token_index).view(
        batch_size, n_strings, -1
    )
    mask = (pos != StringVocabulary.pad_token_index).long()
    operators = torch.tensor(operators)  # batch_size
    return {'pos': pos, 'mask': mask, 'operators': operators}


def set2regex_collate_fn(examples):
    batch_size = len(examples)
    n_strings = len(examples[0]['positive_strings_indices']) + len(examples[0]['negative_strings_indices'])
    strings = []
    types = []
    regex = []
    for example in examples:
        strings.extend(example['positive_strings_indices'] + example['negative_strings_indices'])
        types.extend([0] * len(example['positive_strings_indices']) + [1] * len(example['negative_strings_indices']))
        regex.append(example['regex_indices'])
    strings = pad_sequence([torch.tensor(s) for s in strings], batch_first=True, padding_value=StringVocabulary.pad_token_index).view(
        batch_size, n_strings, -1
    )
    types = torch.tensor(types).view(batch_size, n_strings)
    regex = pad_sequence([torch.tensor(s) for s in regex], batch_first=True, padding_value=RegexVocabulary.pad_token_index).view(
        batch_size, -1
    )
    return {'strings': strings, 'types': types, 'regex': regex}


def add_type_label(example):
    regex = example['regex']
    ast = normalize(regex)
    example['regex_type'] = ast.type
    return example


def add_concat_label(example):
    regex = example['regex']
    ast = normalize(regex)
    example['regex_type'] = ast.type
    if ast.type != 'concat':
        example['labels'] = ['\x00']  # fake label, anyway it will be filtered out
        return example
    positive_strings = example['positive_strings']
    subregexes = []
    for i, child in enumerate(ast.children):
        subregexes.append(f'(?P<G{i}>{child.get_regex()})')  # 0-indexed
    named_regex = ''.join(subregexes)
    labels = []
    for positive_string in positive_strings:
        label = ''
        match = fullmatch(named_regex, positive_string)
        groups = match.groupdict()
        gids = sorted(groups)  # list[str]
        for gid in gids:
            substring = groups[gid]
            if substring is not None:
                gid = chr(int(gid[1:]))  # remove 'G' prefix
                label += len(substring) * gid
        labels.append(label)
    example['labels'] = labels
    return example


def add_union_label(example):
    regex = example['regex']
    ast = normalize(regex)
    example['regex_type'] = ast.type
    if ast.type != 'union':
        example['labels'] = ['\x00']  # fake label, anyway it will be filtered out
        return example
    positive_strings = example['positive_strings']
    subregexes = []
    for i, child in enumerate(ast.children):
        subregexes.append(f'(?P<G{i}>{child.get_regex()})')  # 0-indexed
    named_regex = '|'.join(subregexes)
    labels = []
    for positive_string in positive_strings:
        match = fullmatch(named_regex, positive_string)
        groups = match.groupdict()
        gids = sorted(groups)  # list[str]
        for gid in gids:
            substring = groups[gid]
            if substring is not None:
                idx = gid[1:]  # remove 'G' prefix
                labels.append(idx)
                break
    translation_map = {'0': '\x00'}
    counter = 1
    true_labels = []
    for label in labels:
        if label not in translation_map:
            translation_map[label] = chr(counter)
            counter += 1
        true_labels.append(translation_map[label])
    example['labels'] = true_labels
    return example


def get_dataloader(dataset: Dataset, shuffle, batch_size=64, num_proc=64, mode='segmenter'):
    match mode:
        case 'segmenter':
            add_indices = add_segmenter_indices
            collate_fn = segmenter_collate_fn
            dataset = dataset.map(add_concat_label, num_proc=num_proc)
            dataset = dataset.filter(lambda example: example['regex_type'] == 'concat', num_proc=num_proc)
        case 'partitioner':
            add_indices = add_partitioner_indices
            collate_fn = partitioner_collate_fn
            dataset = dataset.map(add_union_label, num_proc=num_proc)
            dataset = dataset.filter(lambda example: example['regex_type'] == 'union', num_proc=num_proc)
        case 'router':
            dataset = dataset.map(add_type_label, num_proc=num_proc)
            add_indices = add_router_indices
            collate_fn = router_collate_fn
        case 'set2regex':
            add_indices = add_set2regex_indices
            collate_fn = set2regex_collate_fn
    if dataset.num_rows == 0:
        return []
    dataset = dataset.map(add_indices, num_proc=num_proc, with_indices=True)
    if mode != 'segmenter':
        bucket_batcher = BucketBatchSampler(batch_size, dataset['indices_and_lengths'], drop_last=False, shuffle=shuffle)
        dataloader = DataLoader(dataset, collate_fn=collate_fn, num_workers=num_proc, batch_sampler=bucket_batcher)
    else:
        dataloader = DataLoader(dataset, collate_fn=collate_fn, num_workers=num_proc, batch_size=batch_size, shuffle=shuffle)
    return dataloader
