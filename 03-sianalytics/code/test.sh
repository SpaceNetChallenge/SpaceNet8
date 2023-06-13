#!/usr/bin/env bash
DATA_DIR=$1
OUTPUT_FILE=$2
SAVE_DIR='/workspace/out_dir'
LOCATION="/workspace"

FLOOD_CONFIG='flood_1300'
LOUISIANA_BUILDING_CONFIG='build_1024_win12'
LOUISIANA_ROAD_CONFIG='road_1024_aug'

FLOOD_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/flood_submit/${FLOOD_CONFIG}.py"
FLOOD_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${FLOOD_CONFIG}"

LOUISIANA_BUILDING_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/building_submit/${LOUISIANA_BUILDING_CONFIG}.py"
LOUISIANA_BUILDING_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${LOUISIANA_BUILDING_CONFIG}"

LOUISIANA_ROAD_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/road_submit/${LOUISIANA_ROAD_CONFIG}.py"
LOUISIANA_ROAD_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${LOUISIANA_ROAD_CONFIG}"

mkdir -p $SAVE_DIR/out

# Step 0. Download weights
gdown https://drive.google.com/drive/folders/1yFvC5zkM6AmH2i-gia_e9gAaZ-uBwsia --folder -O $LOCATION/mmsegmentation
cd $LOCATION/mmsegmentation && unzip work_dir.zip
cd $LOCATION

# Step 1. Flood Inference
python3 $LOCATION/mmsegmentation/tools/sn8_test.py $FLOOD_CONFIG_PATH "${FLOOD_WORK_DIR}/latest.pth" \
    --image-path "${DATA_DIR}/*/" \
    --out "${SAVE_DIR}/out/flood"

# Step 2. Building Inference
python3 $LOCATION/mmsegmentation/tools/sn8_test.py $LOUISIANA_BUILDING_CONFIG_PATH "${LOUISIANA_BUILDING_WORK_DIR}/latest.pth" \
    --image-path "${DATA_DIR}/*/" \
    --out "${SAVE_DIR}/out/building"

# Step 3. Road Inference
python3 $LOCATION/mmsegmentation/tools/sn8_test.py $LOUISIANA_ROAD_CONFIG_PATH "${LOUISIANA_ROAD_WORK_DIR}/latest.pth" \
    --image-path "${DATA_DIR}/*/" \
    --out "${SAVE_DIR}/out/road/pre" \
    --logit

python3 $LOCATION/processings/merge_prepost_road.py --rootdir $SAVE_DIR/out/road \
    --image-path "${DATA_DIR}/*/" --threshold 100 --preonly

# Step 4. Postprocessing
bash $LOCATION/processings/postprocessing.sh $SAVE_DIR $OUTPUT_FILE
