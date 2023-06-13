FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /
ENV PYTHON_VERSION="3.9.13"
ENV OPENCV_VERSION="4.5.1.48"
ENV GDAL_VERSION="3.0.4.10"
ENV TORCH_VERSION="1.11.0"
ENV TORCHVISION_VERSION="0.12.0"

RUN apt-get update
RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev \
        libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev \
        libbz2-dev wget tar vim libffi-dev curl make libjpeg-dev libtiff-dev \
        unzip yasm pkg-config libswscale-dev libtbb2 libtbb-dev libpng-dev \
        libavformat-dev libpq-dev libproj-dev libsqlite3-dev libexpat1-dev \
        libcurl4-gnutls-dev locales git

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz \
 && tar xvf Python-$PYTHON_VERSION.tar.xz \
 && rm Python-$PYTHON_VERSION.tar.xz \
 && cd Python-$PYTHON_VERSION \
 && ./configure \
 && make -j 8 \
 && make install -j 8

RUN  cd .. \
 && rm -rf Python-$PYTHON_VERSION

RUN pip3 install --upgrade pip

RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN ln -sv /usr/local/bin/pip3 /usr/bin/pip

RUN apt-get install gdal-bin libgdal-dev -y


RUN pip3 install --no-cache-dir wheel cmake numpy==1.22.4 scikit-image pillow yacs \
    cython matplotlib tqdm jupyter lxml fastkml shapely termcolor pyshp \
    opencv-python==$OPENCV_VERSION 

RUN pip3 install --no-cache-dir pygdal==$GDAL_VERSION

RUN pip install torch==${TORCH_VERSION}+cu113 torchvision==${TORCHVISION_VERSION}+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Basic Setup
RUN pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
WORKDIR /workspace
COPY baseline /workspace/baseline
COPY mmsegmentation /workspace/mmsegmentation
COPY processings /workspace/processings
RUN pip install -e /workspace/mmsegmentation

COPY requirements.txt . 
COPY test.sh .
COPY train.sh .
COPY Dockerfile .
RUN pip install -r /workspace/requirements.txt

# download SpaceNet2 Building and SpaceNet3 Road data
# Check /workspace/processings/preprocess_sn2_sn3.py for details
RUN pip install gdown
RUN apt install -y unzip
