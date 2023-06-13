#!/usr/bin/env bash
# Example usage : ./train.sh /data/SN8_flood/train/
TRAIN_DATA_DIR=$1

# MysteryCity_Test_Private_label_image_mapping.csv
# Data preparation
rm -rf /wdata/*
cp -r ${TRAIN_DATA_DIR} /wdata
python baseline/data_prep/geojson_prep.py --root_dir /wdata/train --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public
python baseline/data_prep/create_masks.py --root_dir /wdata/train --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public
python baseline/data_prep/generate_train_val_test_csvs.py --root_dir /wdata/train --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public --out_csv_basename sn8_data --val_percent 0.15 --out_dir /wdata
# Output : /wdata/sn8_data_train.csv, /wdata/sn8_data_val.csv
# NOTE : Additional files for batch must be added in Louisiana-West test set!!! 
# Run Training
torchrun --nproc_per_node=4 train.py --batch-size 8 --epochs 600
