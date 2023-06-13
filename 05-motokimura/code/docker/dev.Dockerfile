FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND="noninteractive" TZ="Asia/Tokyo" apt-get install -y \
    postgresql-client \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# install segmentation_models_pytorch
RUN git clone https://github.com/qubvel/segmentation_models.pytorch.git /smp
WORKDIR /smp
RUN git checkout 740dab561ccf54a9ae4bb5bda3b8b18df3790025 && pip install .

ENV PYTHONPATH $PYTHONPATH:/work

WORKDIR /work
