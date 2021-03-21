FROM tensorflow/tensorflow:latest-gpu-jupyter

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN apt-get clean && apt-get update

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . /lib/connoisseur
RUN cp /lib/connoisseur
RUN python setup.py develop
 
ENV JOBLIB_TEMP_FOLDER /joblib_tmp/

