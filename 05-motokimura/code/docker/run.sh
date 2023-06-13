#!/bin/bash

IMAGE="sn8:dev"
CONTAINER="sn8_dev"

# set project root dicrectory to map to docker
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# set path to directories to map to docker
DATA_DIR=/data/spacenet8
ARTIFACT_DIR=/data/spacenet8_artifact

docker run --runtime nvidia -it --rm --ipc=host \
	-v ${PROJ_DIR}:/work \
	-v ${DATA_DIR}:/data \
	-v ${ARTIFACT_DIR}:/wdata \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
