#!/bin/bash
# example usage: ./scripts/this_script.sh 0 --config path/to/config General.seed=1000

EXP_ID=$1
TEST_ARGS=${@:2}

echo EXP_ID: $EXP_ID
echo TEST_ARGS: $TEST_ARGS

# XXX: 5-fold
python tools/test_net.py --exp_id $(($EXP_ID+0)) $TEST_ARGS
python tools/test_net.py --exp_id $(($EXP_ID+1)) $TEST_ARGS
python tools/test_net.py --exp_id $(($EXP_ID+2)) $TEST_ARGS
python tools/test_net.py --exp_id $(($EXP_ID+3)) $TEST_ARGS
python tools/test_net.py --exp_id $(($EXP_ID+4)) $TEST_ARGS
