#!/bin/bash

IMAGE="sn8:dev"

# get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

cd ${PROJ_DIR} && \
docker build -t ${IMAGE} -f ${THIS_DIR}/dev.Dockerfile .
