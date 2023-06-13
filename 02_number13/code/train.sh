#!/bin/bash
echo "delete previous models..."
rm -rf /wdata/models_sn8
rm -rf /wdata/models_flood
cheak () {
    if [ -f "$1" ]; then
        echo "$1 EXISTS."
    else
        echo "$1 DOES NOT EXIST."
    fi
}
sh ./prep_sn8_data.sh $1
sh ./download_data.sh
#echo "sleep after downloading..."
sleep 100
mkdir -p /root/.cache/torch/hub/checkpoints
#cp /wdata/inceptionresnetv2-520b38e4.pth /root/.cache/torch/hub/checkpoints/
#cp /mdata/se_resnext50_32x4d-a260b3a4.pth /root/.cache/torch/hub/checkpoints/

if [ -d "train_val_csvs" ]
then
  echo "train_val_csvs(path for each folds) exists."
else
  echo "train val splits csv not present creating new..."
  python tools/baseline_data_prep/generate_train_val_test_csvs.py --root_dir /wdata --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public \
            --out_csv_basename sn8_data --val_percent 0.15 --out_dir train_val_csvs
fi

model_out_root="/wdata/"


echo "1#------------------------------------------------------------------"
mkdir -p /root/.cache/torch/hub/checkpoints
model_name_1="inceptionresnetv2"
pretrained_1=$model_out_root"/models_sn_prev/"$model_name_1
out_dir_1=$model_out_root"/models_sn8/"$model_name_1
flood_dir_1=$model_out_root"/models_flood/"$model_name_1
train_csv_1="train_val_csvs/sn8_data_train_fold1.csv"
val_csv_1="train_val_csvs/sn8_data_val_fold1.csv"
cheak $train_csv_1
cheak $val_csv_1
gpu_1=0
wget --no-check-certificate http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth -P /root/.cache/torch/hub/checkpoints/ &&
python train.py --model_name $model_name_1 --out_dir $pretrained_1 \
          --tr previous --gpu $gpu_1 --batch_size 12 &&
python train.py --model_name $model_name_1 --weights $pretrained_1"/previous_"$model_name_1"_last.pth" --out_dir $out_dir_1 \
        --tr foundation --gpu $gpu_1 --train_csvs $train_csv_1 --val_csvs $val_csv_1 &


echo "2#------------------------------------------------------------------"
model_name_2="se_resnext50_32x4d"
mkdir -p /root/.cache/torch/hub/checkpoints
wget --no-check-certificate http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth -P /root/.cache/torch/hub/checkpoints/ &&
pretrained_2=$model_out_root"/models_sn_prev/"$model_name_2
out_dir_2=$model_out_root"/models_sn8/"$model_name_2
flood_dir_2=$model_out_root"/models_flood/"$model_name_2
train_csv_2="train_val_csvs/sn8_data_train_fold2.csv"
val_csv_2="train_val_csvs/sn8_data_val_fold2.csv"
cheak $train_csv_2
cheak $val_csv_2
gpu_2=1
python train.py --model_name $model_name_2 --out_dir $pretrained_2 \
          --tr previous --gpu $gpu_2 --batch_size 12  &&
python train.py --model_name $model_name_2 --weights $pretrained_2"/previous_"$model_name_2"_last.pth" --out_dir $out_dir_2 \
        --tr foundation --gpu $gpu_2 --train_csvs $train_csv_2 --val_csvs $val_csv_2 &


echo "3#------------------------------------------------------------------"
model_name_3="timm-efficientnet-b5"
pretrained_3=$model_out_root"/models_sn_prev/"$model_name_3
out_dir_3=$model_out_root"/models_sn8/"$model_name_3
flood_dir_3=$model_out_root"/models_flood/"$model_name_3
train_csv_3="train_val_csvs/sn8_data_train_fold3.csv"
val_csv_3="train_val_csvs/sn8_data_val_fold3.csv"
cheak $train_csv_3
cheak $val_csv_3
gpu_3=2
python train.py --model_name $model_name_3 --out_dir $pretrained_3 \
          --tr previous --gpu $gpu_3 --batch_size 6 &&
python train.py --model_name $model_name_3 --weights $pretrained_3"/previous_"$model_name_3"_last.pth" --out_dir $out_dir_3 \
        --tr foundation --gpu $gpu_3 --train_csvs $train_csv_3 --val_csvs $val_csv_3 &

echo "4#------------------------------------------------------------------"
model_name_4="resnet50"
pretrained_4=$model_out_root"/models_sn_prev/"$model_name_4
out_dir_4=$model_out_root"/models_sn8/"$model_name_4
flood_dir_4=$model_out_root"/models_flood/"$model_name_4
train_csv_4="train_val_csvs/sn8_data_train_fold4.csv"
val_csv_4="train_val_csvs/sn8_data_val_fold4.csv"
gpu_4=3
cheak $train_csv_4
cheak $val_csv_4
python train.py --model_name $model_name_4 --out_dir $pretrained_4 \
          --tr previous --gpu $gpu_4 --batch_size 12  &&
python train.py --model_name $model_name_4 --weights $pretrained_4"/previous_"$model_name_4"_last.pth" --out_dir $out_dir_4 \
          --tr foundation --gpu $gpu_4 --train_csvs $train_csv_4 --val_csvs $val_csv_4 &&
python train.py --model_name $model_name_4  --weights $out_dir_4"/foundation_"$model_name_4"_last.pth"   --out_dir $flood_dir_4 \
        --tr flood --gpu $gpu_4 --train_csvs configs/sn8_data_train_all.csv --val_csvs $val_csv_4

wait
echo "Spacenet previous models cheak..."

cheak $pretrained_1"/previous_"$model_name_1"_last.pth"
cheak $pretrained_2"/previous_"$model_name_2"_last.pth"
cheak $pretrained_3"/previous_"$model_name_3"_last.pth"
cheak $pretrained_4"/previous_"$model_name_4"_last.pth"

echo "Spacenet 8 foundation models cheak..."

cheak $out_dir_1"/foundation_"$model_name_1"_last.pth"
cheak $out_dir_2"/foundation_"$model_name_2"_last.pth"
cheak $out_dir_3"/foundation_"$model_name_3"_last.pth"
cheak $out_dir_4"/foundation_"$model_name_4"_last.pth"

echo "Spacenet 8 flood models cheak..."
cheak $flood_dir_4"/flood_"$model_name_4"_last.pth"

