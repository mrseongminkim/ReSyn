from dataclasses import dataclass


@dataclass
class StringConfig:
    max_repeat: int = 20
    max_n_strings: int = 10
    max_string_length: int = 110
    duplicate_factor: int = 1
    MAXREPEAT: int = 1_000  # parse.cc; static int maximum_repeat_count = 1000;


@dataclass
class Set2RegexConfig:
    max_regex_length: int = 110


@dataclass
class RouterConfig:
    n_concat: int = 435881
    n_union: int = 46656
    n_no_op: int = 927743


@dataclass
class CheckPointConfig:
    router: str = '20260511_143827_best_loss.pt'
    set2regex: str = '20260511_143851_best_loss.pt'
    partitioner: str = '20260511_143604_best_loss.pt'
    segmenter: str = '20260511_160550_best_loss.pt'


@dataclass
class TimeoutConfig:
    synthesis: int = 10
    fa_equivalence: int = 1
