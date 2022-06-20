#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=NCBI
SEED=42
EPOCH=25
PRETRAIN_EPOCH=5
HMM_LR=0.01
NN_LR=0.001
TRANS_NN_WEIGHT=1
EMISS_NN_WEIGHT=1
BATCH_SIZE=64
N_WORKERS=0

# majority voting
CUDA_VISIBLE_DEVICES=1 python universal_test.py \
  --dataset "$DATASET" \
  --test_task majority \
  --random_seed "$SEED" \
  --update_src_data

# hidden markov model with em
CUDA_VISIBLE_DEVICES=1 python universal_test.py \
  --dataset "$DATASET" \
  --test_task hmm \
  --epoch "$EPOCH" \
  --random_seed "$SEED" \
  --update_src_data \
  --debugging_mode

# neural HMM without transition probabilities
CUDA_VISIBLE_DEVICES=5 python universal_test.py \
  --dataset "$DATASET" \
  --test_task iid \
  --trans_nn_weight "$TRANS_NN_WEIGHT" \
  --emiss_nn_weight "$EMISS_NN_WEIGHT" \
  --epoch "$EPOCH" \
  --pretrain_epoch "$PRETRAIN_EPOCH" \
  --hmm_lr "$HMM_LR" \
  --nn_lr "$NN_LR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$N_WORKERS" \
  --random_seed "$SEED" \
  --update_src_data \
  --pin_memory \
  --debugging_mode

# neural HMM
CUDA_VISIBLE_DEVICES=5 python universal_test.py \
  --dataset "$DATASET" \
  --test_task nhmm \
  --trans_nn_weight "$TRANS_NN_WEIGHT" \
  --emiss_nn_weight "$EMISS_NN_WEIGHT" \
  --epoch "$EPOCH" \
  --pretrain_epoch "$PRETRAIN_EPOCH" \
  --hmm_lr "$HMM_LR" \
  --nn_lr "$NN_LR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$N_WORKERS" \
  --random_seed "$SEED" \
  --update_src_data \
  --pin_memory \
  --debugging_mode
