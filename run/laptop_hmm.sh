#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=Laptop
TASK=hmm
EPOCH=25
SEED=42

CUDA_VISIBLE_DEVICES=1 python universal_test.py \
  --dataset "$DATASET" \
  --test_task "$TASK" \
  --epoch "$EPOCH" \
  --random_seed "$SEED" \
  --update_src_data
