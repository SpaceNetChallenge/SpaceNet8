#!/bin/bash
sh ./download_models.sh
solution_out=$2
cheak () {
    if [ -f "$1" ]; then
        echo "$1 EXISTS."
    else
        echo "$1 DOES NOT EXIST."
    fi
}
CONDA_RUN="conda run --no-capture-output -n gdal"
subdir=$(find $1 -maxdepth 1 -mindepth 1 -type d)
out_dir="/wdata/"$(basename -- $subdir)"/"
mkdir -p $out_dir
main_preds_dir=$out_dir"/preds_main"
mkdir -p $main_preds_dir
cp -r $subdir /wdata/
cp $1"/resolutions.txt" $out_dir
$CONDA_RUN python tools/convert_gdalwarp_postimg.py $out_dir/"POST-event" $out_dir"/POST-event-warped"

mapping=$(find $out_dir -name "*mapping.csv")
echo $mapping

echo "1#------------------------------------------------------------------"
model_name_1="inceptionresnetv2"
pred_dir_1=$out_dir"/preds_"$model_name_1
model_weights_1="/wdata/models_sn8/"$model_name_1"/foundation_"$model_name_1"_last.pth"
cheak $model_weights_1
python inference.py --test_dir_pre $out_dir"/PRE-event/" --pred_dir  $pred_dir_1 --model_path $model_weights_1 --model_name $model_name_1 \
                    --task both  --gpu 0 --tta &

echo "2#------------------------------------------------------------------"
model_name_2="se_resnext50_32x4d"
pred_dir_2=$out_dir"/preds_"$model_name_2
model_weights_2="/wdata/models_sn8/"$model_name_2"/foundation_"$model_name_2"_last.pth"
cheak $model_weights_2
python inference.py --test_dir_pre $out_dir"/PRE-event/" --pred_dir  $pred_dir_2 --model_path $model_weights_2 --model_name $model_name_2 \
                    --task both  --gpu 1 --tta &

echo "3#------------------------------------------------------------------"
model_name_3="timm-efficientnet-b5"
pred_dir_3=$out_dir"/preds_"$model_name_3
model_weights_3="/wdata/models_sn8/"$model_name_3"/foundation_"$model_name_3"_last.pth"
cheak $model_weights_3
python inference.py --test_dir_pre $out_dir"/PRE-event/" --pred_dir  $pred_dir_3 --model_path $model_weights_3 --model_name $model_name_3 \
                   --task both --gpu 2 --tta &

echo "4#------------------------------------------------------------------"
model_name_4="resnet50"
pred_dir_4=$out_dir"/preds_"$model_name_4
model_weights_4="/wdata/models_sn8/"$model_name_4"/foundation_"$model_name_4"_last.pth"
siam_weights_4="/wdata/models_flood/"$model_name_4"/flood_"$model_name_4"_last.pth"
cheak $model_weights_4
cheak $siam_weights_4
python inference.py --test_dir_pre $out_dir"/PRE-event/" --pred_dir  $pred_dir_4 --model_path $model_weights_4 --model_name $model_name_4 \
                    --task both --gpu 3 --tta &&
python inference.py --test_dir_pre $out_dir"/PRE-event/" --test_dir_post $out_dir"/POST-event-warped/" --mapping_csv $mapping \
                 --pred_dir  $main_preds_dir --model_path $siam_weights_4 --model_name $model_name_4 --gpu 3 --flood --thres 0.3

wait

python ens_preds.py --masks $pred_dir_4 $pred_dir_2 $pred_dir_3 $pred_dir_1 --out_dir $main_preds_dir --thres 0.5
road_wkt="road_wkt.csv"
building_wkt="building_wkt.csv"
$CONDA_RUN python tools/postprocessing/buildings/building_postprocessing.py --im_dir $out_dir"/PRE-event/" \
           --foundation_pred_dir $main_preds_dir  --flood_pred_dir $main_preds_dir --out_submission_csv $building_wkt
$CONDA_RUN python tools/postprocessing/roads/vectorize_roads.py --pred_dir $main_preds_dir --out_csv $road_wkt
python ens_preds.py --postprocess_wkt --build_wkt $building_wkt --road_wkt $road_wkt --solution_out $solution_out
echo "done"

