#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2

foundation=$(printf "%05d" "$foundation")
flood=$(printf "%05d" "$flood")

road_dir=/wdata/_val/road_submissions/exp_${foundation}_${flood}
building_dir=/wdata/_val/building_submissions/exp_${foundation}_${flood}

python tools/submit.py --val --road $road_dir --building $building_dir
