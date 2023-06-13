FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Install required python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN conda init bash
RUN conda create -n gdal -c conda-forge python=3.8 gdal fiona shapely scikit-image pandas
RUN conda run -n gdal pip install opencv-python numba
RUN pip install imagecodecs
RUN conda run -n gdal  pip install tqdm geopandas
RUN pip install awscli
RUN conda run -n gdal  pip install osmnx
RUN pip install torchsampler
RUN apt-get install wget

COPY . ./
RUN chmod 777 download_data.sh
RUN chmod 777 download_models.sh
RUN chmod 777 train.sh
RUN chmod 777 test.sh
RUN ls -la ./
