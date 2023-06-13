#!/usr/bin/env bash
DATA_DIR=$1 
GPUS=4
LOCATION="/workspace"

FLOOD_CONFIG='flood_1300'
ROAD_CONFIG='road_1024_aug'
BUILDING_CONFIG='build_1024_win12'

SN3_CONFIG='road_pretrained'
SN2_CONFIG='build_pretrained'

FLOOD_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/flood_submit/${FLOOD_CONFIG}.py"
FLOOD_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${FLOOD_CONFIG}"

SN3_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/road_submit/${SN3_CONFIG}.py"
SN3_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${SN3_CONFIG}"

SN2_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/building_submit/${SN2_CONFIG}.py"
SN2_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${SN2_CONFIG}"

ROAD_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/road_submit/${ROAD_CONFIG}.py"
ROAD_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${ROAD_CONFIG}"

BUILDING_CONFIG_PATH="${LOCATION}/mmsegmentation/configs/building_submit/${BUILDING_CONFIG}.py"
BUILDING_WORK_DIR="${LOCATION}/mmsegmentation/work_dir/${BUILDING_CONFIG}"


# Step 0 Preprocess data
mkdir -p /data/SpaceNet8/

# generate training data
bash $LOCATION/processings/preprocessing.sh $DATA_DIR

cd $LOCATION/mmsegmentation

# Step 1 Flood Training
PORT=29500
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/train.py \
    $FLOOD_CONFIG_PATH \
    --work-dir $FLOOD_WORK_DIR \
    --launcher pytorch \
    --no-validate

# Step 2.0. Download data
gdown https://drive.google.com/drive/folders/1w1-oX6rHy7ouF1MIg24e88xTW2MuUSIK?usp=sharing --folder -O /data
cd /data && unzip pretraining.zip
cd $LOCATION/mmsegmentation

# Step 2 SN3 Training
PORT=29501
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/train.py \
    $SN3_CONFIG_PATH \
    --work-dir $SN3_WORK_DIR \
    --launcher pytorch \
    --no-validate

# Step 3 SN2 Training
PORT=29502
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/train.py \
    $SN2_CONFIG_PATH \
    --work-dir $SN2_WORK_DIR \
    --launcher pytorch \
    --no-validate

#### Step 4 SN8 Road Training
PORT=29503
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/train.py \
    $ROAD_CONFIG_PATH \
    --work-dir $ROAD_WORK_DIR \
    --launcher pytorch \
    --load-from "${SN3_WORK_DIR}/latest.pth" \
    --no-validate

#### Step 5 SN8 Building Training
PORT=29504
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/train.py \
    $BUILDING_CONFIG_PATH \
    --work-dir $BUILDING_WORK_DIR \
    --launcher pytorch \
    --load-from "${SN2_WORK_DIR}/latest.pth" \
    --no-validate
