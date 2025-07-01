FROM continuumio/miniconda3:4.8.3-alpine

USER root

RUN apk update && apk upgrade && apk add bash

COPY environment.yml /tmp/environment.yml

RUN /usr/sbin/addgroup -S askcos && \
    /usr/sbin/adduser -D -u 1000 askcos -G askcos && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/askcos/.profile && \
    echo "conda activate base" >> /home/askcos/.profile

RUN /opt/conda/bin/conda env update --name base --file /tmp/environment.yml && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

USER askcos

COPY --chown=askcos:askcos . /home/askcos/wln-keras-fw

WORKDIR /home/askcos/wln-keras-fw

ENV PATH=/opt/conda/bin${PATH:+:${PATH}}
ENV PYTHONPATH=/home/askcos/wln-keras-fw:/opt/conda/share/RDKit/Contrib${PYTHONPATH:+:${PYTHONPATH}}

LABEL dock.maintainer.name="MLPDS developers"
LABEL dock.maintainer.email="mlpds@mit.edu"
