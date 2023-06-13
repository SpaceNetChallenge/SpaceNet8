#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2

foundation_dir=$(printf "/wdata/_val/preds/exp_%05d" "$foundation")
flood_dir=$(printf "/wdata/_val/preds/exp_%05d" "$flood")
python tools/building/postproc_building.py --val --foundation $foundation_dir --flood $flood_dir
