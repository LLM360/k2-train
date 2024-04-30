# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""
import os.path

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
import datasets

N_CHUNKS = 360


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(
        keys=['token_ids'], data=data, datatype=torch.int64)

    tokens = data_b['token_ids'].long()
    labels = torch.ones_like(tokens)
    labels[..., :-1] = tokens[..., 1:]
    tokens, labels = tokens.contiguous(), labels.contiguous()

    assert not any([
        args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss])

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss)

    loss_mask[..., -1] = 0

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = (
        get_batch(data_iterator))
    timers('batch-generator').stop()

    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


# TODO: use path and split portion provided in the data args.
def my_train_valid_test_datasets_provider(num_samples):
    args = get_args()

    print_rank_0('building datasets using huggingface datasets...')

    latest_ckpt_iter = int(open(f'{args.save}/latest_checkpointed_iteration.txt').read().strip())
    chunk_begin_idx = latest_ckpt_iter // args.save_interval
    print_rank_0(f'chunk_begin_idx = {chunk_begin_idx}')
    
#     while True:
#         ckpt_dir = \
#             f'{args.save}/iter_{(chunk_begin_idx + 1) * args.save_interval:07d}'
#         if not os.path.exists(ckpt_dir):
#             print_rank_0(f'chunk_begin_idx = {chunk_begin_idx}')
#             break
#         else:
#             chunk_begin_idx += 1
#             print_rank_0(
#                 f'{ckpt_dir} exists. chunk_idx chanced to {chunk_begin_idx}.')
    
    chunk_idxes = (
            list(range(16)) + list(range(32, 32 + 6)) + list(range(44, 44 + 8)))
    chunk_idxes.extend(
        [i for i in range(360) if i not in chunk_idxes])
    chunk_idxes = chunk_idxes[:160] + chunk_idxes[161:] + chunk_idxes[160:161]
    chunk_idxes = chunk_idxes[:185] + chunk_idxes[186:] + chunk_idxes[185:186]
    
    chunk_idxes = chunk_idxes[:args.n_chunks]
    print_rank_0(f'chunk_idxes: {chunk_idxes}')
    print_rank_0(f'actual chunk_idxes this run: {chunk_idxes[chunk_begin_idx:]}')

    data_files = [
        f"{args.data_base_path}/chunk_{chunk_idx}.jsonl"
        for chunk_idx in chunk_idxes
    ]
    train_ds = datasets.load_dataset(
        "json",
        data_files=data_files,
        split='train',
        num_proc=min(args.n_chunks, os.cpu_count()),
        cache_dir='/mount/data/train_cache')
    train_ds = train_ds.with_format("np")

    print_rank_0(f'huggingface dataset built, size = {len(train_ds)}')

    valid_ds, test_ds = None, None
    print_rank_0("> finished creating pretrain datasets ...")
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(
        train_valid_test_dataset_provider=my_train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        args_defaults={'tokenizer_type': 'LLaMATokenizer'})