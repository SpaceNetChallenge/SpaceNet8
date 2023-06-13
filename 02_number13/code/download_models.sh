#!/bin/bash
echo "Downloading model weights...."
get_models_sn8 () {
    if [ -f "$1" ]; then
        echo "EXISTS."
    else
    	wget https://www.dropbox.com/s/2eoruek40n5hpfj/models_sn8_local.tar.gz?dl=1 -O /wdata/models_sn8.tar.gz
        #wget https://www.dropbox.com/s/3dg4qvd2c0pyj2k/models_sn8.tar.gz?dl=1 -O /wdata/models_sn8.tar.gz
        tar -xvzf /wdata/models_sn8.tar.gz -C /wdata
    fi
}
get_models_flood () {
    if [ -f "$1" ]; then
        echo "EXISTS."
    else
        wget https://www.dropbox.com/s/o6io0342yb352q3/models_flood.tar.gz?dl=1 -O /wdata/models_flood.tar.gz
        tar -xvzf /wdata/models_flood.tar.gz -C /wdata
    fi
}
get_models_sn8 /wdata/models_sn8/resnet50/foundation_resnet50_last.pth
get_models_flood /wdata/models_flood/resnet50/flood_resnet50_last.pth



