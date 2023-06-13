#!/bin/bash

cd /wdata

# --- Download External Data ---
aws s3 cp s3://zaburo-sn8-data/xview2/xview2_geotiff.tgz . &

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_${AOI}.tar.gz . &
done

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_${AOI}.tar.gz . &
    aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_${AOI}_geojson_roads_speed.tar.gz . &
done

for AOI in AOI_7_Moscow AOI_8_Mumbai; do
    aws s3 cp s3://spacenet-dataset/spacenet/SN5_roads/tarballs/SN5_roads_train_${AOI}.tar.gz . &
done
wait

# --- Extract ---
tar -xvzf xview2_geotiff.tgz
rm -rf xview2_geotiff.tgz

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    tar -xvzf SN2_buildings_train_${AOI}.tar.gz &
done
wait
for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    rm -rf SN2_buildings_train_${AOI}.tar.gz
done
for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    rm -rf ${AOI}_Train/MUL
    rm -rf ${AOI}_Train/PAN
    rm -rf ${AOI}_Train/RGB-PanSharpen
done

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    tar -xvzf SN3_roads_train_${AOI}.tar.gz &
    tar -xvzf SN3_roads_train_${AOI}_geojson_roads_speed.tar.gz &
done
wait 
for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    rm -rf SN3_roads_train_${AOI}.tar.gz
    rm -rf SN3_roads_train_${AOI}_geojson_roads_speed.tar.gz
done
for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    rm -rf ${AOI}/MS
    rm -rf ${AOI}/PAN
    rm -rf ${AOI}/PS-RGB
done

for AOI in AOI_7_Moscow AOI_8_Mumbai; do
    tar -xvzf SN5_roads_train_${AOI}.tar.gz &
done
wait
for AOI in AOI_7_Moscow AOI_8_Mumbai; do
    rm -rf SN5_roads_train_${AOI}.tar.gz
done
for AOI in AOI_7_Moscow AOI_8_Mumbai; do
    rm -rf ${AOI}/MS
    rm -rf ${AOI}/PAN
    rm -rf ${AOI}/PS-RGB
done
mv nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload/train/AOI_7_Moscow .
mv nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload/train/AOI_8_Mumbai .
rm -rf nfs
