#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import argparse
import csv
import json
import logging
import os
import pathlib
import pickle
import sys
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import (
        add_encoder_params, 
        setup_args_gpu, 
        print_args, 
        set_encoder_params_from_state,
        add_tokenizer_params, 
        add_cuda_params,
        )
from dpr.utils.data_utils import (
    Tensorizer,
    read_ctxs
)
from dpr.utils.model_utils import (
        setup_for_distributed_mode, 
        get_model_obj, 
        load_states_from_checkpoint,
        move_to_device
        )


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def gen_ctx_vectors(args, ctx_rows: List[dict], model: nn.Module, tensorizer: Tensorizer,
                    insert_title: bool = True) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):

        batch_token_tensors = [tensorizer.text_to_tensor(ctx['text'], title=ctx['title'] if insert_title else None) for ctx in
                               ctx_rows[batch_start:batch_start + bsz]]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0),args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch),args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch),args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r['id'] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy())
            for i in range(out.size(0))
        ])

        if total % 10 == 0:
            logger.info('Encoded passages %d', total)

    return results


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Generate dense embeddings from Wikipedia')
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--output_dir', required=True, type=str, default=None,
                        help='output .tsv file path to write results to ')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'
    setup_args_gpu(args)
    
    # model load 
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)
    
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model
    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)

    logger.info('reading data from file=%s', args.ctx_file)
    rows = read_ctxs(args.ctx_file)

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]

    data = gen_ctx_vectors(args, rows, encoder, tensorizer, True)

    fo_emb = "{dest}/emb_{model}{suffix}.pickle".format(
        dest = args.output_dir,
        model = os.path.basename(args.model_file).replace('.pt', ''),
        suffix = f'_{args.shard_id}' if args.num_shards>1 else '',
    )
    pathlib.Path(os.path.dirname(fo_emb)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % fo_emb)
    with open(fo_emb, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), fo_emb)

    

if __name__ == '__main__':
    main()

