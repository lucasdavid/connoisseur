FROM tensorflow/tensorflow:nightly-gpu-py3

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN pip install -r ../docs/requirements-base.txt --upgrade --no-cache-dir

WORKDIR "/"

ENV JOBLIB_TEMP_FOLDER /joblib_tmp/
