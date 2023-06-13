#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate sn8

# A sample call to your testing script follows. Again, folder and file names are for example only, you should not assume that the exact same names will be used in testing.
# ./test.sh /data/SN8_flood/test/ /wdata/my_output.csv
# In this sample case the testing data looks like this:
#   data/
#     SN8_flood/
#       test/
#         Louisiana-West_Test_Public/
#           POST-event/
#           PRE-event/
#           Louisiana-West_Test_Public_label_image_mapping.csv
#         resolutions.txt

test_dir=$(find $1 -name '*_label_image_mapping.csv' -type f -print -quit | xargs dirname)
output_path=$2

cd /root/

if [ ! -e /wdata/working/roads/efficientnetv2m-junction-01/0/final.pt ]; then
    mkdir -p /wdata/working/
    rsync -av /root/models/ /wdata/working/
fi

for fold_index in 0 1 2 3; do
    python scripts/roads/train.py predict_test \
        --config_path scripts/roads/configs/efficientnetv2m-junction-01.yaml \
        --image_dir ${test_dir}/PRE-event \
        training.device="cuda:${fold_index}" \
        fold_index=${fold_index} \
        --d4_tta True &
done
wait

for fold_index in 0 1 2 3; do
    python scripts/buildings/train.py predict_test \
        --config_path scripts/buildings/configs/efficientnetv2s.yaml \
        --image_dir ${test_dir}/PRE-event \
        training.device="cuda:${fold_index}" \
        fold_index=${fold_index} \
        --d4_tta True &
done
wait

for fold_index in 0 1 2 3; do
    python scripts/floods/train.py predict_test \
        --config_path scripts/floods/configs/resnet50-cj-dice-half-flag-xview2.yaml \
        --image_dir ${test_dir} \
        training.device="cuda:${fold_index}" \
        fold_index=${fold_index} \
        --d4_tta True &
done
wait

python scripts/postprocess/average_predictions.py \
    --out_dir /wdata/working/roads/efficientnetv2m-junction-01/avg_pred_test \
    /wdata/working/roads/efficientnetv2m-junction-01/0/pred_test \
    /wdata/working/roads/efficientnetv2m-junction-01/1/pred_test \
    /wdata/working/roads/efficientnetv2m-junction-01/2/pred_test \
    /wdata/working/roads/efficientnetv2m-junction-01/3/pred_test

python scripts/postprocess/average_predictions.py \
    --out_dir /wdata/working/buildings/efficientnetv2s/avg_pred_test \
    /wdata/working/buildings/efficientnetv2s/0/pred_test/ \
    /wdata/working/buildings/efficientnetv2s/1/pred_test/ \
    /wdata/working/buildings/efficientnetv2s/2/pred_test/ \
    /wdata/working/buildings/efficientnetv2s/3/pred_test/

python scripts/postprocess/average_predictions.py \
    --out_dir /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/avg_pred_test \
    /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/0/pred_test \
    /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/1/pred_test \
    /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/2/pred_test \
    /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/3/pred_test

find /wdata/working/roads/efficientnetv2m-junction-01/avg_pred_test -name '*.tif' -print -type f \
    | xargs -I {} basename {} .tif \
    | xargs -I {} -P16 python scripts/postprocess/road_postprocess.py \
        --mask_path /wdata/working/roads/efficientnetv2m-junction-01/avg_pred_test/{}.tif \
        --out_dir /wdata/working/roads/efficientnetv2m-junction-01/avg_pred_test_post \
        --flood_mask /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/avg_pred_test/{}.tif \
        --speed_conversion_file /root/roads_train_speed_conversion_binned.csv \
        --extension_length_pix 50 \
        --road_skeleton_th 0.6 \
        --small_isolate_path_length_pix 400

find /wdata/working/buildings/efficientnetv2s/avg_pred_test -name '*.tif' -print -type f \
    | xargs -I {} basename {} .tif \
    | xargs -I {} -P16 python scripts/postprocess/building_postprocess.py \
        --mask_path /wdata/working/buildings/efficientnetv2s/avg_pred_test/{}.tif \
        --out_dir /wdata/working/buildings/efficientnetv2s/avg_pred_test_post \
        --flood_mask /wdata/working/floods/resnet50-cj-dice-half-flag-xview2/avg_pred_test/{}.tif \
        --simplify_tolerance 0.25

python scripts/postprocess/merge_dfs.py \
    --data_dir ${test_dir} \
    /wdata/working/roads/efficientnetv2m-junction-01/avg_pred_test_post \
    /wdata/working/buildings/efficientnetv2s/avg_pred_test_post \
    ${output_path}
