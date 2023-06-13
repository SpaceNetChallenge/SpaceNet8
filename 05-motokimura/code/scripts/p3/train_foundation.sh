#!/bin/bash
# example usage: ./scripts/this_script.sh 0 0 0 --config path/to/config General.seed=1000

TASK=foundation
GPU_ID=$1
FOLD_ID=$2
EXP_ID=$3
TRAIN_ARGS=${@:4}

echo TASK: $TASK
echo GPU_ID: $GPU_ID
echo FOLD_ID: $FOLD_ID
echo EXP_ID: $EXP_ID
echo TRAIN_ARGS: $TRAIN_ARGS

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train_net.py --task $TASK --exp_id $EXP_ID --fold_id $FOLD_ID $TRAIN_ARGS
