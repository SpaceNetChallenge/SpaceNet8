FROM nvcr.io/nvidia/pytorch:22.03-py3
MAINTAINER Jacob Arndt <arndtjw@ornl.gov>

RUN apt-get update --fix-missing && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install postgresql-client libpq-dev gdal-bin libgdal-dev curl -y

WORKDIR /tmp

COPY requirements.txt .

RUN set -eux && pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt
