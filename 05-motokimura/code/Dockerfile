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

# download selimsef's xView2 models (published under Apache 2.0 license)
# https://github.com/selimsef/xview2_solution
# https://github.com/selimsef/xview2_solution/releases/tag/0.0.1
WORKDIR /work/selimsef_xview2_models
RUN wget -nv https://github.com/selimsef/xview2_solution/releases/download/0.0.1/localization_densenet_unet_densenet161_3_0_best_dice
RUN wget -nv https://github.com/selimsef/xview2_solution/releases/download/0.0.1/localization_densenet_unet_densenet161_3_1_best_dice
RUN wget -nv https://github.com/selimsef/xview2_solution/releases/download/0.0.1/pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview
RUN wget -nv https://github.com/selimsef/xview2_solution/releases/download/0.0.1/pseudo_densenet_seamese_unet_shared_densenet161_2_best_xview

# download XD_XD's SpaceNet-5 winning models
# see docs/prepare_sn5_XD_XD_models.md to know how I prepared these models
WORKDIR /work/xdxd_sn5_models
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/xdxd_sn5_serx50_focal.zip
RUN unzip xdxd_sn5_serx50_focal.zip && rm -f xdxd_sn5_serx50_focal.zip

# download motokimura's home-built models
# these models are removed before training (see train.sh)
WORKDIR /work/models
# effnet-b5 foundation
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50000.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50001.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50002.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50003.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50004.zip
# effnet-b5 flood
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50010.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50011.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50012.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50013.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_50014.zip
# effnet-b6 foundation
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60400.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60401.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60402.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60403.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60404.zip
# effnet-b6 flood
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60420.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60421.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60422.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60423.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_60424.zip
# se-resnext50 foundation (road only, finetuned from XD_XD's unet)
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80000.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80001.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80002.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80003.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80004.zip
# densenet161 foundation (building only, finetuned from selimsef's unet)
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80500.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80501.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80502.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80503.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_80504.zip
# densenet161 flood (building only, finetuned from selimsef's siamese unet)
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_90010.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_90011.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_90012.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_90013.zip
RUN wget -nv https://motokimura-public-sn8.s3.amazonaws.com/exp_90014.zip
RUN unzip "*.zip" && rm -f *.zip

# copy files
COPY configs /work/configs
COPY spacenet8_model /work/spacenet8_model
COPY tools /work/tools
COPY test.sh /work/
COPY train.sh /work/

RUN chmod a+x /work/test.sh
RUN chmod a+x /work/train.sh

ENV PYTHONPATH $PYTHONPATH:/work

WORKDIR /work
