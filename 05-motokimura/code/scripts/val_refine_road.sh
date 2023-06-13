#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2
road=$3

foundation_dir=$(printf "/wdata/_val/preds/exp_%05d" "$foundation")
flood_dir=$(printf "/wdata/_val/preds/exp_%05d" "$flood")
road_dir=$(printf "/wdata/_val/preds/exp_%05d" "$road")

refined_dir=$(printf "/wdata/_val/refined_preds/exp_%05d" "$foundation")
vector_dir=$(printf "/wdata/_val/road_vectors/exp_%05d" "$foundation")
graph_dir=$(printf "/wdata/_val/road_graphs/exp_%05d" "$foundation")

python tools/refine_road_mask.py --val --foundation $foundation_dir --road $road_dir
python tools/road/vectorize.py --val --foundation $refined_dir
python tools/road/to_graph.py --val --vector $vector_dir
python tools/road/insert_flood.py --val --graph $graph_dir --flood $flood_dir
