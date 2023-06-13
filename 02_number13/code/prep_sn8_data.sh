#!/bin/bash
CONDA_RUN="conda run --no-capture-output -n gdal"
echo "/data is read only but we need to write annotation fixes and create mask so we copy all to /wdata...dataset is small!"
mkdir -p /wdata/Germany_Training_Public
cp -r $1"/Germany_Training_Public" /wdata/
mkdir -p /wdata/Louisiana-East_Training_Public
cp -r $1"/Louisiana-East_Training_Public" /wdata/
cp $1/"resolutions.txt" /wdata

$CONDA_RUN python tools/baseline_data_prep/geojson_prep.py --root_dir /wdata/ --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public
$CONDA_RUN python tools/baseline_data_prep/create_masks.py --root_dir /wdata \
                  --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

$CONDA_RUN python tools/convert_gdalwarp_postimg.py /wdata/Germany_Training_Public/POST-event /wdata/Germany_Training_Public/POST-event-warped
$CONDA_RUN python tools/convert_gdalwarp_postimg.py /wdata/Louisiana-East_Training_Public/POST-event /wdata/Louisiana-East_Training_Public/POST-event-warped

python tools/baseline_data_prep/generate_train_val_test_csvs.py --root_dir /wdata  \
      --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public \
      --out_csv_basename sn8_data --val_percent 0.15 --out_dir train_val_csvs