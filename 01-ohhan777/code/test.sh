#!/usr/bin/env bash
# Example usage : ./test.sh /data/SN8_flood/test/ /wdata/my_output.csv
TEST_DIR=$1
output_path=$2
WEIGHTS_DIR="./runs/train/weights" # the directory holding foundation road prediction .tifs. they have suffix _roadspeedpred.tif
ROAD_PRED_DIR="./runs/best" # the directory holding foundation road prediction .tifs. they have suffix _roadspeedpred.tif
FLOOD_PRED_DIR="./runs/best" # the directory holding flood prediction .tifs. They have suffix _floodpred.tif
torchrun --nproc_per_node=4 val_best.py --batch-size 8 --dataset-path ${TEST_DIR} --weights-dir ${WEIGHTS_DIR} --task test --tiffs 

OUT_SKNW_WKT="${ROAD_PRED_DIR}/sknw_wkt.csv"
GRAPH_NO_SPEED_DIR="${ROAD_PRED_DIR}/graphs_nospeed"
WKT_TO_G_LOG_FILE="${ROAD_PRED_DIR}/wkt_to_G.log"

GRAPH_SPEED_DIR="${ROAD_PRED_DIR}/graphs_speed"
INFER_SPEED_LOG_FILE="${ROAD_PRED_DIR}/graph_speed.log"

ROAD_SUBMISSION_CSV_FILEPATH="${ROAD_PRED_DIR}/MysteryCity_roads_submission.csv" # the name of the submission .csv
BUILDING_SUBMISSION_CSV_FILEPATH="${ROAD_PRED_DIR}/MysteryCity_buildings_submission.csv"
SUBMISSION_CSV_FILEPATH=${output_path}
FLOOD_OUTPUT_SHAPEFILE_PATH="${ROAD_PRED_DIR}/flood_road_speed.shp"
BUILD_OUTPUT_SHAPEFILE_PATH="${ROAD_PRED_DIR}/build_shps"

python baseline/postprocessing/roads/vectorize_roads.py --im_dir $ROAD_PRED_DIR --out_dir $ROAD_PRED_DIR --write_shps --write_graphs --write_csvs --write_skeletons

python baseline/postprocessing/roads/wkt_to_G.py --wkt_submission $OUT_SKNW_WKT --graph_dir $GRAPH_NO_SPEED_DIR --log_file $WKT_TO_G_LOG_FILE --min_subgraph_length_pix 20 --min_spur_length_m 10

python baseline/postprocessing/roads/infer_speed.py --eval_csv ${TEST_DIR} --mask_dir $ROAD_PRED_DIR --graph_dir $GRAPH_NO_SPEED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --log_file $INFER_SPEED_LOG_FILE
 
python baseline/postprocessing/roads/create_submission_test.py --flood_pred $FLOOD_PRED_DIR --graph_speed_dir $GRAPH_SPEED_DIR --output_csv_path $ROAD_SUBMISSION_CSV_FILEPATH --output_shapefile_path $FLOOD_OUTPUT_SHAPEFILE_PATH

python baseline/postprocessing/buildings/building_postprocessing_test.py --foundation_pred_dir $ROAD_PRED_DIR --flood_pred_dir $FLOOD_PRED_DIR --out_submission_csv $BUILDING_SUBMISSION_CSV_FILEPATH --out_shapefile_dir $BUILD_OUTPUT_SHAPEFILE_PATH --square_size 2 --simplify_tolerance 0.3 --min_area 5 --percent_positive 0.5

sed -i '1d' $ROAD_SUBMISSION_CSV_FILEPATH
cat $BUILDING_SUBMISSION_CSV_FILEPATH $ROAD_SUBMISSION_CSV_FILEPATH > $SUBMISSION_CSV_FILEPATH

python baseline/postprocessing/more_filter_submission.py --in_csv_path $SUBMISSION_CSV_FILEPATH --output_csv_path $SUBMISSION_CSV_FILEPATH
