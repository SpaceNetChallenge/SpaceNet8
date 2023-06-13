#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2

foundation_dir=$(printf "/wdata/_val/preds/exp_%05d" "$foundation")
flood_dir=$(printf "/wdata/_val/preds/exp_%05d" "$flood")

vector_dir=$(printf "/wdata/_val/road_vectors/exp_%05d" "$foundation")
graph_dir=$(printf "/wdata/_val/road_graphs/exp_%05d" "$foundation")

python tools/road/vectorize.py --val --foundation $foundation_dir
python tools/road/to_graph.py --val --vector $vector_dir
python tools/road/insert_flood.py --val --graph $graph_dir --flood $flood_dir
