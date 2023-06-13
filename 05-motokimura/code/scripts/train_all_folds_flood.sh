#!/bin/bash
# example usage: ./scripts/this_script.sh 1000 0 --config path/to/config General.seed=1000

TASK=flood
EXP_ID=$1
PRETRAINED_ID=$2
TRAIN_ARGS=${@:3}

echo TASK: $TASK
echo EXP_ID: $EXP_ID
echo PRETRAINED_ID: $PRETRAINED_ID
echo TRAIN_ARGS: $TRAIN_ARGS

# XXX: 5-fold
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+0)) --fold_id 0 --pretrained $(($PRETRAINED_ID+0)) $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+1)) --fold_id 1 --pretrained $(($PRETRAINED_ID+1)) $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+2)) --fold_id 2 --pretrained $(($PRETRAINED_ID+2)) $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+3)) --fold_id 3 --pretrained $(($PRETRAINED_ID+3)) $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+4)) --fold_id 4 --pretrained $(($PRETRAINED_ID+4)) $TRAIN_ARGS
