#!/bin/bash
# example usage: ./scripts/this_script.sh 0 --config path/to/config General.seed=1000

EXP_ID=$1
TEST_ARGS=${@:2}

./scripts/test_all_folds.sh $EXP_ID --val $TEST_ARGS