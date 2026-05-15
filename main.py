import logging
import os
import random
import sys
import warnings

import numpy as np
import torch

from baselines.gpt import GPTSynthesizer
from baselines.oss import GPTOssSynthesizer
from baselines.prax import Prax
from baselines.trivial import TrivialSynthesizer
from ReSyn.server import Set2RegexServer
from utils.evaluator import Evaluator
from utils.splitters import (
    FallbackSplitter,
    FixedRecursiveSplitter,
    ForestSplitter,
    OracleSplitter,
    PartitionerOnlySplitter,
    ReSynSplitter,
    SegmenterOnlySplitter,
)


def ensure_determinism(seed=42):
    warnings.filterwarnings('ignore')
    current_seed = os.environ.get('PYTHONHASHSEED')
    seed_string = str(seed)
    if current_seed is None or current_seed != seed_string:
        os.environ['PYTHONHASHSEED'] = seed_string
        os.execl(sys.executable, sys.executable, *sys.argv)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


ensure_determinism()

warnings.filterwarnings('ignore')


def _get_synthesizer(synthesizer='prax', do_sample=False, beam_size=0):
    match synthesizer:
        case 'prax':
            return Prax(do_sample=do_sample)
        case 'set2regex':
            return Set2RegexServer(do_sample=do_sample, beam_size=beam_size)
        case 'gpt':
            return GPTSynthesizer()
        case 'oss':
            return GPTOssSynthesizer()
        case 'trivial':
            return TrivialSynthesizer()
        case _:
            raise ValueError(f'Unknown synthesizer: {synthesizer}')


def _add_splitter(synthesizer, split_mode):
    synthesizer.fallback = True
    match split_mode:
        case 'oracle':
            return OracleSplitter(synthesizer)
        case 'resyn':
            return ReSynSplitter(synthesizer)
        case 'forest':
            return ForestSplitter(synthesizer)
        case 'segmenter':
            return SegmenterOnlySplitter(synthesizer)
        case 'partitioner':
            return PartitionerOnlySplitter(synthesizer)
        case 'fallback':
            return FallbackSplitter(synthesizer)
        case 'fixed':
            return FixedRecursiveSplitter(synthesizer)
        case 'baseline':
            synthesizer.fallback = False
            return synthesizer
        case _:
            raise ValueError(f'Unknown splitter: {split_mode}')


def evaluate(synthesizer, splitter, split_by, gpu_id, deduple_by_signature=True, beam_size=0):
    os.makedirs('logs/experiments', exist_ok=True)
    logger = logging.getLogger('Experiments')
    logging.basicConfig(
        filename=f'logs/experiments/{synthesizer}_{splitter}_{split_by}.log'
        if not beam_size
        else f'logs/experiments/{synthesizer}_{splitter}_{split_by}_{beam_size}.log',
        encoding='utf-8',
        level=logging.INFO,
        filemode='a',
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        force=True,
    )
    torch.cuda.set_device(gpu_id)
    base_synthesizer = _get_synthesizer(synthesizer, beam_size=beam_size)
    synthesizer_with_splitter = _add_splitter(base_synthesizer, splitter)
    evaluator = Evaluator(synthesizer_with_splitter, logger, split_by, deduple_by_signature)
    evaluator.evaluate()


if __name__ == '__main__':
    torch.cuda.set_device(2)
    for synthesizer in (
        'prax',
        'set2regex',
    ):
        for splitter in (
            'baseline',
            'forest',
            'segmenter',
            'partitioner',
            'oracle',
            'fixed',
            'resyn',
        ):
            base_synthesizer = _get_synthesizer(synthesizer)
            synthesizer_with_splitter = _add_splitter(base_synthesizer, splitter)
            evaluator = Evaluator(synthesizer_with_splitter, None, 'source', True)
            evaluator.evaluate_and_save_all(f'logs/experiments_pickles/{synthesizer}_{splitter}.pkl')
