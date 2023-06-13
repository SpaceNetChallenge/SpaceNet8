DATA_DIR=$1 
LOCATION="/workspace"
SPACENET8_TOSAVE_DIR="/data/SpaceNet8"

# Generate Masks
python3 $LOCATION/baseline/data_prep/geojson_prep.py --root_dir $DATA_DIR \
    --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

python3 $LOCATION/baseline/data_prep/create_masks.py --root_dir $DATA_DIR \
    --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

# generate csv
python3 $LOCATION/baseline/data_prep/generate_train_val_test_csvs.py --root_dir $DATA_DIR \
    --out_dir $SPACENET8_TOSAVE_DIR --out_csv_basename train \
    --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public \
    --val_percent 0

# generate mmstyle datasets
python3 $LOCATION/processings/preprocess_data.py --csv $SPACENET8_TOSAVE_DIR/train.csv