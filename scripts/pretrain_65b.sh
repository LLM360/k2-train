#!/bin/bash


bcprun -p 8 --env 'CUDA_DEVICE_MAX_CONNECTIONS=1' -c 'python /mount/training/k2_training/main.py \
--position-embedding-type rope \
--normalization RMSNorm \
--disable-bias-linear \
--swiglu \
--num-layers 80 \
--hidden-size 8192 \
--ffn-hidden-size 22016 \
--num-attention-heads 64 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--layernorm-epsilon 1e-5 \
--untie-embeddings-and-output-weights \
--use-flash-attn \
--vocab-size 32032 \
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 4 \
--sequence-parallel \
--micro-batch-size 4 \
--global-batch-size 2040 \
--data-base-path /mount/data/shuffled_data_chunks \
--n-chunks 360 \
--train-iters 338760 \
--save-interval 941 \
--lr 1.5e-4 \
--lr-decay-style cosine \
--min-lr 1.5e-5 \
--weight-decay 0.1 \
--lr-warmup-iters 2000 \
--clip-grad 1.0 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--hidden-dropout 0.0 \
--attention-dropout 0.0 \
--bf16 \
--tensorboard-log-interval 1 \
--wandb-proj-name k2_official_from160 \
--log-interval 1 \
--tensorboard-dir /mount/training/k2_training/tensorboard-llama65b \
--distributed-timeout-minutes 600 \
--save /mount/ckpts/llama-65b-mp \
--load /mount/ckpts/llama-65b-mp \
--seed 22238
'
