#!/bin/bash
# |-ROOT_PRED_DIR
# | |-foundation
# | |-flood

EVAL_CSV="/data/datasets/SpaceNet8/Training/distributed_csv/Loisiana-West_test.csv" # the .csv that prediction was run on
ROOT_PRED_DIR="/data/workspace/doyoungi/SpaceNet8/output/submit_0819"
LOCATION="/data/workspace/doyoungi/test/spacenet8/baseline/postprocessing"

ROAD_PRED_DIR="${ROOT_PRED_DIR}/foundation" # the directory holding foundation road prediction .tifs. they have suffix _roadspeedpred.tif
FLOOD_PRED_DIR="${ROOT_PRED_DIR}/flood" # the directory holding flood prediction .tifs. They have suffix _floodpred.tif

### ROAD
OUT_SKNW_WKT="${ROAD_PRED_DIR}/sknw_wkt.csv"
GRAPH_NO_SPEED_DIR="${ROAD_PRED_DIR}/graphs_nospeed"
WKT_TO_G_LOG_FILE="${ROAD_PRED_DIR}/wkt_to_G.log"

GRAPH_SPEED_DIR="${ROAD_PRED_DIR}/graphs_speed"
INFER_SPEED_LOG_FILE="${ROAD_PRED_DIR}/graph_speed.log"

SUBMISSION_CSV_FILEPATH="${ROOT_PRED_DIR}/MysteryCity_roads_submission.csv" # the name of the submission .csv
OUTPUT_SHAPEFILE_PATH="${ROAD_PRED_DIR}/flood_road_speed.shp"

python $LOCATION/roads/vectorize_roads.py --im_dir $ROAD_PRED_DIR --out_dir $ROAD_PRED_DIR --write_shps --write_graphs --write_csvs --write_skeletons

python $LOCATION/roads/wkt_to_G.py --wkt_submission $OUT_SKNW_WKT --graph_dir $GRAPH_NO_SPEED_DIR --log_file $WKT_TO_G_LOG_FILE --min_subgraph_length_pix 20 --min_spur_length_m 10

python $LOCATION/roads/infer_speed.py --eval_csv $EVAL_CSV --mask_dir $ROAD_PRED_DIR --graph_dir $GRAPH_NO_SPEED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --log_file $INFER_SPEED_LOG_FILE
 
python $LOCATION/roads/create_submission.py --flood_pred $FLOOD_PRED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --output_csv_path $SUBMISSION_CSV_FILEPATH --output_shapefile_path $OUTPUT_SHAPEFILE_PATH


### Building
BUILDING_SUBMISSION_CSV_FILEPATH="${ROOT_PRED_DIR}/building_submission.csv"
SHPFILE_FILEPATH="${ROOT_PRED_DIR}/building_shapefile"
SOLUTION_CSV_FILEPATH="${ROOT_PRED_DIR}/solution.csv"

mkdir $SHPFILE_FILEPATH

python $LOCATION/buildings/building_postprocessing.py --foundation_pred_dir $ROAD_PRED_DIR \
    --flood_pred_dir $FLOOD_PRED_DIR --out_submission_csv $BUILDING_SUBMISSION_CSV_FILEPATH \
    --out_shapefile_dir $SHPFILE_FILEPATH --square_size 5 --simplify_tolerance 0.30 \
    --min_area 5 --percent_positive 0.25

python $LOCATION/merge_csvs.py --road-csv $SUBMISSION_CSV_FILEPATH --bldg-csv $BUILDING_SUBMISSION_CSV_FILEPATH --out $SOLUTION_CSV_FILEPATH

