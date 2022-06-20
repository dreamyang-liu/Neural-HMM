#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=Laptop
DATA_DIR='../data/Laptop'
TASK=majority
SEED=42

CUDA_VISIBLE_DEVICES=1 python ../universal_test.py \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --test_task "$TASK" \
  --random_seed "$SEED" \
  --update_src_data
