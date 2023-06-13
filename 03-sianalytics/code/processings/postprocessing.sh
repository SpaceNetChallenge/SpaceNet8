#!/bin/bash
# |-SAVE_DIR
# | |-foundation
# | |-flood
# | |-out
# | | |-building
# | | |-road
# | | |-flood

SAVE_DIR=$1
OUTPUT_FILE=$2
EVAL_CSV="/goodeuljang" # the .csv that prediction was run on
LOCATION="/workspace"

ROAD_PRED_DIR="${SAVE_DIR}/foundation" # the directory holding foundation road prediction .tifs. they have suffix _roadspeedpred.tif
FLOOD_PRED_DIR="${SAVE_DIR}/flood" # the directory holding flood prediction .tifs. They have suffix _floodpred.tif

mkdir -p $ROAD_PRED_DIR


## Prepare foundation
# BUILDING
python3 $LOCATION/processings/floodpred_to_buildingpred.py --rootdir $SAVE_DIR/out/building \
    --outdir $SAVE_DIR/foundation --classes 2 --building
# ROAD
python3 $LOCATION/processings/floodpred_to_buildingpred.py --rootdir $SAVE_DIR/out/road \
    --outdir $SAVE_DIR/foundation --classes 2 --road
# FLOOD
ln -s $SAVE_DIR/out/flood $FLOOD_PRED_DIR


## Post processing
# ROAD
OUT_SKNW_WKT="${ROAD_PRED_DIR}/sknw_wkt.csv"
GRAPH_NO_SPEED_DIR="${ROAD_PRED_DIR}/graphs_nospeed"
WKT_TO_G_LOG_FILE="${ROAD_PRED_DIR}/wkt_to_G.log"

GRAPH_SPEED_DIR="${ROAD_PRED_DIR}/graphs_speed"
INFER_SPEED_LOG_FILE="${ROAD_PRED_DIR}/graph_speed.log"

SUBMISSION_CSV_FILEPATH="${SAVE_DIR}/MysteryCity_roads_submission.csv" # the name of the submission .csv
OUTPUT_SHAPEFILE_PATH="${ROAD_PRED_DIR}/flood_road_speed.shp"


python3 $LOCATION/baseline/postprocessing/roads/vectorize_roads_cannab.py --rootdir $ROAD_PRED_DIR --outdir $ROAD_PRED_DIR

python3 $LOCATION/baseline/postprocessing/roads/wkt_to_G.py --wkt_submission $OUT_SKNW_WKT \
    --graph_dir $GRAPH_NO_SPEED_DIR --log_file $WKT_TO_G_LOG_FILE \
    --min_subgraph_length_pix 20 --min_spur_length_m 10

python3 $LOCATION/baseline/postprocessing/roads/infer_speed.py --eval_csv $EVAL_CSV \
    --mask_dir $ROAD_PRED_DIR --graph_dir $GRAPH_NO_SPEED_DIR \
    --graph_speed_dir $GRAPH_SPEED_DIR --log_file $INFER_SPEED_LOG_FILE
 
python3 $LOCATION/baseline/postprocessing/roads/create_submission.py --flood_pred $FLOOD_PRED_DIR \
    --graph_speed_dir $GRAPH_SPEED_DIR --output_csv_path $SUBMISSION_CSV_FILEPATH \
    --output_shapefile_path $OUTPUT_SHAPEFILE_PATH

# Building
BUILDING_SUBMISSION_CSV_FILEPATH="${SAVE_DIR}/building_submission.csv"
SHPFILE_FILEPATH="${SAVE_DIR}/building_shapefile"
SOLUTION_CSV_FILEPATH="${SAVE_DIR}/solution.csv"

mkdir -p $SHPFILE_FILEPATH

python3 $LOCATION/baseline/postprocessing/buildings/building_postprocessing.py --foundation_pred_dir $ROAD_PRED_DIR \
    --flood_pred_dir $FLOOD_PRED_DIR --out_submission_csv $BUILDING_SUBMISSION_CSV_FILEPATH \
    --out_shapefile_dir $SHPFILE_FILEPATH --square_size 5 --simplify_tolerance 0.30 \
    --min_area 5 --percent_positive 0.25

python3 $LOCATION/baseline/postprocessing/merge_csvs.py --road-csv $SUBMISSION_CSV_FILEPATH \
    --bldg-csv $BUILDING_SUBMISSION_CSV_FILEPATH --out $SOLUTION_CSV_FILEPATH

mkdir -p `dirname ${OUTPUT_FILE}` && cp -f $SOLUTION_CSV_FILEPATH $OUTPUT_FILE

echo "Hello World!"