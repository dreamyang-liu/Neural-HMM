#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

TASK=true-trans
SEED=42
EPOCH=20
PRETRAIN_EPOCH=5
HMM_LR=0.01
NN_LR=0.0001
TRANS_NN_WEIGHT=1
EMISS_NN_WEIGHT=1
BATCH_SIZE=100
N_WORKERS=0

CUDA_VISIBLE_DEVICES=7 python conll_test.py \
  --test_task "$TASK" \
  --trans_nn_weight "$TRANS_NN_WEIGHT" \
  --emiss_nn_weight "$EMISS_NN_WEIGHT" \
  --epoch "$EPOCH" \
  --pretrain_epoch "$PRETRAIN_EPOCH" \
  --hmm_lr "$HMM_LR" \
  --nn_lr "$NN_LR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$N_WORKERS" \
  --random_seed "$SEED" \
  --pre_train \
  --pin_memory \
  --debugging_mode
