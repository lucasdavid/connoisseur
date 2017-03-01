FROM tensorflow/tensorflow:nightly-gpu-py3

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN pip install --upgrade --no-cache-dir numpy \
                                         scipy \
                                         scikit-learn \
                                         h5py \
                                         keras \
                                         image \
                                         sacred==0.7b2 \
                                         opencv-python \
                                         tensorflow-gpu

WORKDIR "/"

ENV JOBLIB_TEMP_FOLDER /joblib_tmp/
