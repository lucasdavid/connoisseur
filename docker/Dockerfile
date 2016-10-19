FROM tensorflow/tensorflow:nightly-gpu-py3

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN pip --no-cache-dir install --upgrade \
    h5py \
    pymongo \
    opencv-python \
    sacred \
    scikit-learn \
    keras \
    image

WORKDIR "/"

ENV JOBLIB_TEMP_FOLDER /joblib_tmp/
