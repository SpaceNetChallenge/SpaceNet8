#!/bin/bash
# example usage: ./scripts/this_script.sh 0 0 1000 0 --config path/to/config General.seed=1000

TASK=flood
GPU_ID=$1
FOLD_ID=$2
EXP_ID=$3
PRETRAINED_ID=$4
TRAIN_ARGS=${@:5}

echo TASK: $TASK
echo GPU_ID: $GPU_ID
echo FOLD_ID: $FOLD_ID
echo EXP_ID: $EXP_ID
echo PRETRAINED_ID: $PRETRAINED_ID
echo TRAIN_ARGS: $TRAIN_ARGS

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train_net.py --task $TASK --exp_id $EXP_ID --pretrained $PRETRAINED_ID --fold_id $FOLD_ID $TRAIN_ARGS
