#!/bin/bash

CONDA_RUN="conda run --no-capture-output -n gdal"
out_dir="/wdata/spacenet-dataset/"
mkdir -p $out_dir
# replace with "--dryrun" to-do actual dryrun
dryrun='--no-sign-request'
out_dir=$out_dir"/"

object="buildings/"
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_4_Shanghai/PS-RGB $out_dir$object"/AOI_4_Shanghai/PS-RGB" --recursive --no-sign-request $dryrun
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_3_Paris/PS-RGB $out_dir$object"/AOI_3_Paris/PS-RGB" --recursive --no-sign-request $dryrun
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB $out_dir$object"/AOI_2_Vegas/PS-RGB" --recursive --no-sign-request $dryrun
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_5_Khartoum/PS-RGB $out_dir$object"/AOI_5_Khartoum/PS-RGB" --recursive --no-sign-request $dryrun

for VARIABLE in $out_dir$object"/AOI_4_Shanghai/PS-RGB" $out_dir$object"/AOI_3_Paris/PS-RGB" \
                $out_dir$object"/AOI_2_Vegas/PS-RGB" $out_dir$object"/AOI_5_Khartoum/PS-RGB"
	do
    $CONDA_RUN  python tools/create_uint8.py --in_dir $VARIABLE --out_dir $VARIABLE"-u8"
  done

object="roads/"
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/train/AOI_2_Vegas/PS-RGB $out_dir$object"/AOI_2_Vegas/PS-RGB" --recursive --no-sign-request $dryrun
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/train/AOI_3_Paris/PS-RGB $out_dir$object"/AOI_3_Paris/PS-RGB" --recursive --no-sign-request $dryrun
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/train/AOI_4_Shanghai/PS-RGB $out_dir$object"/AOI_4_Shanghai/PS-RGB" --recursive --no-sign-request $dryrun
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/train/AOI_5_Khartoum/PS-RGB $out_dir$object"/AOI_5_Khartoum/PS-RGB" --recursive --no-sign-request $dryrun

for VARIABLE in $out_dir$object"/AOI_4_Shanghai/PS-RGB" $out_dir$object"/AOI_3_Paris/PS-RGB" \
                $out_dir$object"/AOI_2_Vegas/PS-RGB" $out_dir$object"/AOI_5_Khartoum/PS-RGB"
	do
    $CONDA_RUN  python tools/create_uint8.py --in_dir $VARIABLE --out_dir $VARIABLE"-u8"
  done

#already uint8
aws s3 cp s3://spacenet-dataset/spacenet/SN5_roads/train/AOI_8_Mumbai/PS-RGB/ $out_dir$object"/AOI_8_Mumbai/PS-RGB/" --recursive --no-sign-request  $dryrun
mv $out_dir$object"AOI_8_Mumbai/PS-RGB/" $out_dir$object"AOI_8_Mumbai/PS-RGB-u8"

#already uint8
aws s3 cp s3://spacenet-dataset/spacenet/SN5_roads/train/AOI_7_Moscow/PS-RGB/ $out_dir$object"/AOI_7_Moscow/PS-RGB/" --recursive --no-sign-request $dryrun
mv $out_dir$object"/AOI_7_Moscow/PS-RGB/" $out_dir$object"/AOI_7_Moscow/PS-RGB-u8/"

#Get all buildings Labels
object="buildings/"
for VARIABLE in "AOI_4_Shanghai/geojson_buildings" "AOI_3_Paris/geojson_buildings" \
                "AOI_2_Vegas/geojson_buildings" "AOI_5_Khartoum/geojson_buildings"
	do
    aws s3 cp "s3://spacenet-dataset/spacenet/SN2_buildings/train/"$VARIABLE $out_dir$object$VARIABLE --recursive --no-sign-request $dryrun
  done

for VARIABLE in "AOI_4_Shanghai/" "AOI_3_Paris/" \
                "AOI_2_Vegas/" "AOI_5_Khartoum/"
	do
    $CONDA_RUN python tools/create_building_masks.py $out_dir$object$VARIABLE
  done

object="roads/"
for VARIABLE in "AOI_4_Shanghai/" "AOI_3_Paris/" \
                "AOI_2_Vegas/" "AOI_5_Khartoum/"
	do
    aws s3 cp "s3://spacenet-dataset/spacenet/SN3_roads/train/"$VARIABLE"geojson_roads_speed" $out_dir$object$VARIABLE"geojson_roads_speed" \
            --recursive --no-sign-request $dryrun
    $CONDA_RUN python tools/create_road_masks.py --path_data $out_dir$object$VARIABLE --output_mask_path $out_dir$object$VARIABLE"/masks"
  done

for VARIABLE in "AOI_7_Moscow/" "AOI_8_Mumbai/"
	do
    aws s3 cp "s3://spacenet-dataset/spacenet/SN5_roads/train/"$VARIABLE"geojson_roads_speed" $out_dir$object$VARIABLE"geojson_roads_speed" \
            --recursive --no-sign-request $dryrun
    $CONDA_RUN python tools/create_road_masks.py --path_data $out_dir$object$VARIABLE --output_mask_path $out_dir$object$VARIABLE"/masks" --is_SN5
  done

echo "Removing uint16 data asthey are no longer needed..."
object="buildings/"
rm -rf $out_dir$object"/AOI_4_Shanghai/PS-RGB"
rm -rf $out_dir$object"/AOI_3_Paris/PS-RGB"
rm -rf $out_dir$object"/AOI_2_Vegas/PS-RGB"
rm -rf $out_dir$object"/AOI_5_Khartoum/PS-RGB"
object="roads/"
rm -rf $out_dir$object"/AOI_2_Vegas/PS-RGB"
rm -rf $out_dir$object"/AOI_3_Paris/PS-RGB"
rm -rf $out_dir$object"/AOI_4_Shanghai/PS-RGB"
rm -rf $out_dir$object"/AOI_5_Khartoum/PS-RGB"