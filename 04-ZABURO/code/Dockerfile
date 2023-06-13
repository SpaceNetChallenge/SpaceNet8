FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# To avoid apt install stopping with timezone configuration
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    cmake \
    parallel \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
    emacs \
    tmux \
    language-pack-ja-base \
    language-pack-ja \
    zsh \
    less \
    gawk \
    htop \
    unar \
    libproj-dev proj-data proj-bin libgeos-dev \
    default-jdk \
    rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN umask 000 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    chmod -R 777 /opt/conda

RUN umask 000 && \
    conda config --remove channels defaults && \
    conda config --add channels conda-forge && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels intel && \
    conda config --prepend channels pytorch && \
    conda config --prepend channels plotly

COPY environment.yaml /root/
RUN umask 000 && \
    conda env create -f ~/environment.yaml && \
    conda clean -afy && \
    chmod -R 777 /opt/conda

ENV PATH /opt/conda/envs/sn8/bin:$PATH

# AWS CLI for PFS support
ENV AWS_EC2_METADATA_DISABLED=true
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip && \
    cd /tmp/ && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm -r aws awscliv2.zip && \
    aws --version

COPY . /root/
WORKDIR /root/
RUN mv /root/aws /root/.aws
RUN chmod a+x /root/train.sh
RUN chmod a+x /root/test.sh

RUN pip install -e /root/submodules/somen --no-deps

ENV PYTHONPATH .
ENV MKL_NUM_THREADS 1
ENV OMP_NUM_THREADS 1
ENV NUMEXPR_NUM_THREADS 1
RUN python scripts/roads/train.py download --config_path scripts/roads/configs/efficientnetv2m-junction-01.yaml \
    && python scripts/buildings/train.py download --config_path scripts/buildings/configs/efficientnetv2s.yaml \
    && python scripts/floods/train.py download --config_path scripts/floods/configs/resnet50-cj-dice-half-flag-xview2.yaml

RUN aws s3 sync s3://zaburo-sn8-data/models /root/models
