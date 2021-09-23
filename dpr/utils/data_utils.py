#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""

import gzip
import sys
import json
import logging
import math
import pickle
import random
from typing import List, Iterator, Callable

from torch import Tensor as T

logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

END = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            data = pickle.load(reader)
            results.extend(data)
            logger.info(f'| READ ... L-{len(results)} {path}')
    logger.info(f'| SIZE Total data ... {len(results)}')
    return results


def read_data_from_json_files(paths: List[str], upsample_rates: List = None) -> List:
    results = []
    if upsample_rates is None:
        upsample_rates = [1] * len(paths)
    assert len(upsample_rates) == len(paths), 'up-sample rates parameter doesn\'t match input files amount'
    for i, path in enumerate(paths):
        with gzip.open(path, 'rt') if path.endswith('.json.gz') else open(path, 'r', encoding="utf-8") as f:
            logger.info('Reading file %s' % path)
            data = json.load(f)
            upsample_factor = int(upsample_rates[i]) # 1
            data = data * upsample_factor
            results.extend(data)
            logger.info(f'| READ ... L-{len(results)} {path}')
    return results


def read_qas(qa_file):
    questions, question_answers = [], []
    assert qa_file.endswith((
        '.csv', '.tsv', '.json', '.jsonl',
        '.csv.gz', '.tsv.gz', '.json.gz', '.jsonl.gz'
    ))
    logger.info('Reading file %s' % qa_file)
    with gzip.open(qa_file, 'rt') if qa_file.endswith('.gz') else open(qa_file) as fi:
        if qa_file.endswith(('.csv', '.tsv', '.csv.gz', '.tsv.gz')):
            raise NotImplementedError('')
        elif qa_file.endswith(('.jsonl', '.jsonl.gz')):
            for line in fi:
                line = json.loads(line.strip())
                questions.append(line['question'])
                question_answers.append(line['answers'])
        elif qa_file.endswith(('.json', '.json.gz')):
            for d in json.load(fi):
                questions.append(d['question'])
                question_answers.append(d['answers'])
        else:
            logger.warning('Cannot read qa_file')
    return questions, question_answers

def read_ctxs(ctxs_file, return_dict=False):
    rows = dict() if return_dict else []
    assert ctxs_file.endswith((
        '.csv', '.tsv', '.json', '.jsonl',
        '.csv.gz', '.tsv.gz', '.json.gz', '.jsonl.gz'
    ))
    logger.info('Reading file %s' % ctxs_file)
    with gzip.open(ctxs_file, 'rt') if ctxs_file.endswith('.gz') else open(ctxs_file) as fi:
        if ctxs_file.endswith(('.csv', '.tsv', '.csv.gz', '.tsv.gz')):
            sep = '\t' if ctxs_file.endswith('.tsv') else ','
            for i, line in enumerate(fi):
                line = line.strip().split(sep)
                if i == 0:
                    keys = line
                assert len(line) == len(keys)
                if return_dict:
                    rows[line[keys.index('id')]] = {k:v for k,v in zip(keys, line) if k != 'id'}
                else:
                    rows.append({k:v for k,v in zip(keys, line)})
        elif ctxs_file.endswith(('.jsonl', '.jsonl.gz')):
            for line in fi:
                line = json.loads(line.strip())
                if return_dict:
                    id = line.pop('id')
                    rows[id] = line
                else:
                    rows.append(line)
        elif ctxs_file.endswith(('.json', '.json.gz')):
            if return_dict:
                for d in json.load(fi):
                    id = d.pop('id')
                    rows[id] = d
            else:
                rows = json.load(fi)
        else:
            logger.warning('Cannot read ctxs_file')
    return rows


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """
    def __init__(self, data: list, shard_id: int = 0, num_shards: int = 1, batch_size: int = 1, shuffle=True,
                 shuffle_seed: int = 0, offset: int = 0,
                 strict_batch_size: bool = False
                 ):

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.debug(
            'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d', samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.data)

    def iterate_data(self, epoch: int = 0, is_retriever: bool = True) -> Iterator[List]:
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iterations
            logger.info(f'| SHUFFLE ... iterate_data')
            epoch_rnd: random.Random = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(self.data)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations

        max_iterations = self.max_iterations - self.iteration

        shard_samples = self.data[self.shard_start_idx:self.shard_end_idx]
        for i in range(self.iteration * self.batch_size, len(shard_samples), self.batch_size):
            items = shard_samples[i:i+self.batch_size]
            if is_retriever:
                items = self.reset_negatives(items)
            if self.strict_batch_size and len(items) < self.batch_size:
                logger.debug('Extending batch to max size')
                items.extend(shard_samples[0:self.batch_size - len(items)])
            self.iteration += 1
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            batch = shard_samples[0:self.batch_size]
            yield batch

        logger.debug('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0

    # My Extension: Shuffle 後 Negative がバッチ内から選択されるように reset
    def reset_negatives(self, items):
        """ items[0]: len(items) == batch_size
        {
            'question': str,
            'answers': List[str],
            'positive_ctxs': [{'title':str, 'text':str}],
            'negative_ctxs': [{'title':str, 'text':str}],
            'hard_negative_ctxs': [{'title':str, 'text':str}],
        }
        """
        output = items.copy()
        for i, item in enumerate(items):
            negatives, other_answers = [], []
            for tmp in items:
                if item != tmp:
                    other_answers.extend([t['title'] for t in tmp['positive_ctxs']])
                    negatives.extend(tmp['positive_ctxs'])
            assert all(map(lambda x: x['title'] in other_answers, negatives)), \
                "NegativeCtxsError: 他の answers が正しく negative_ctxs に含まれない"
            output[i]['negative_ctxs'] = negatives

        return output

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data[self.shard_start_idx:self.shard_end_idx]:
            visitor_func(sample)

    def reset_negatives(self, items):
        """ items[0]: len(items) == batch_size
        {
            'question': str,
            'answers': List[str],
            'positive_ctxs': [{'title':str, 'text':str}],
            'negative_ctxs': [{'title':str, 'text':str}],
            'hard_negative_ctxs': [{'title':str, 'text':str}],
        }
        """
        output = items.copy()
        for i, src in enumerate(items):
            negatives, negative_answers = [], []
            for tgt in items:
                if src != tgt:
                    negative_answers.extend([t['title'] for t in tgt['positive_ctxs']])
                    negatives.extend(tgt['positive_ctxs'])
            if not all(neg['title'] in negative_answers for neg in negatives):
                raise ValueError('No answer contain negative_ctxs')
            output[i]['negative_ctxs'] = negatives

        return output


def normalize_question(question: str) -> str:
    if question[-1] == '?':
        question = question[:-1]
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError
