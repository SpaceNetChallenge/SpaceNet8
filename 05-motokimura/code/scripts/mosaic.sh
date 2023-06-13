#!/bin/bash
# example usage: ./scripts/this_script.sh

echo 'mosaicing.. this will take ~20 mins'
echo 'you can check progress from /wdata/mosaics/train_*.txt'

mkdir -p /wdata/mosaics/

nohup python tools/mosaic.py --fold_id 0 > /wdata/mosaics/train_0.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 1 > /wdata/mosaics/train_1.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 2 > /wdata/mosaics/train_2.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 3 > /wdata/mosaics/train_3.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 4 > /wdata/mosaics/train_4.txt 2>&1 &

wait
