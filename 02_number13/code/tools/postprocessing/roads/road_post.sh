#!/bin/bash

EVAL_CSV="" # the .csv that prediction was run on
ROAD_PRED_DIR="" # the directory holding foundation road prediction .tifs. they have suffix _roadspeedpred.tif
FLOOD_PRED_DIR="" # the directory holding flood prediction .tifs. They have suffix _floodpred.tif

OUT_SKNW_WKT="${ROAD_PRED_DIR}/sknw_wkt.csv"
GRAPH_NO_SPEED_DIR="${ROAD_PRED_DIR}/graphs_nospeed"
WKT_TO_G_LOG_FILE="${ROAD_PRED_DIR}/wkt_to_G.log"

GRAPH_SPEED_DIR="${ROAD_PRED_DIR}/graphs_speed"
INFER_SPEED_LOG_FILE="${ROAD_PRED_DIR}/graph_speed.log"

SUBMISSION_CSV_FILEPATH="${ROAD_PRED_DIR}/MysteryCity_roads_submission.csv" # the name of the submission .csv
OUTPUT_SHAPEFILE_PATH="${ROAD_PRED_DIR}/flood_road_speed.shp"

python vectorize_roads.py --im_dir $ROAD_PRED_DIR --out_dir $ROAD_PRED_DIR --write_shps --write_graphs --write_csvs --write_skeletons

python wkt_to_G.py --wkt_submission $OUT_SKNW_WKT --graph_dir $GRAPH_NO_SPEED_DIR --log_file $WKT_TO_G_LOG_FILE --min_subgraph_length_pix 20 --min_spur_length_m 10

python infer_speed.py --eval_csv $EVAL_CSV --mask_dir $ROAD_PRED_DIR --graph_dir $GRAPH_NO_SPEED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --log_file $INFER_SPEED_LOG_FILE
 
python create_submission.py --flood_pred $FLOOD_PRED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --output_csv_path $SUBMISSION_CSV_FILEPATH --output_shapefile_path $OUTPUT_SHAPEFILE_PATH