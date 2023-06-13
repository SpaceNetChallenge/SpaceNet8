#!/bin/bash
# example usage: ./scripts/this_script.sh 0 --config path/to/config General.seed=1000

TASK=foundation
EXP_ID=$1
TRAIN_ARGS=${@:2}

echo TASK: $TASK
echo EXP_ID: $EXP_ID
echo TRAIN_ARGS: $TRAIN_ARGS

# XXX: 5-fold
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+0)) --fold_id 0 $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+1)) --fold_id 1 $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+2)) --fold_id 2 $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+3)) --fold_id 3 $TRAIN_ARGS
python tools/train_net.py --task $TASK --exp_id $(($EXP_ID+4)) --fold_id 4 $TRAIN_ARGS
