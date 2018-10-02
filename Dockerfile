FROM tensorflow/tensorflow:nightly-gpu-py3

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN apt-get clean && apt-get update

RUN apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install python3-dev python3-tk -y

RUN pip install --upgrade --no-cache-dir numpy \
                                         scipy \
                                         scikit-learn \
                                         scikit-image \
                                         h5py \
                                         keras \
                                         h5py \
                                         image \
                                         sacred \
                                         opencv-python \
                                         pandas


ENV JOBLIB_TEMP_FOLDER /joblib_tmp/

WORKDIR "/"
