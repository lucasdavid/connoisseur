# Connoisseur

A few experiments on Machine Learning applied to authorship recognition in
paintings.

## Installation

### Virtualenv

The most basic installation (might run into a couple of incompatibilities)
is through virtualenv.

```shell
$ cd /path/to/connoisseur

$ virtualenv .env --python=/usr/bin/python3
$ source .env/bin/activate

$ python setup.py install
```

### Docker

This method is preferable, as docker containers can improve on stability and
reproducibility given the nature of its containers. I've included a image
that already contains all dependencies (e.g. tensorflow, opencv-python).
It can be build with:

```shell
$ cd /path/to/connoisseur
$ cd docs/
$ docker build -t connoisseur-image .
```

The run the container and install connoisseur:

```shell
$ nvidia-docker run -it -v /mnt/datasets:/datasets \
                        -v /home/ldavid/repos:/repos \
                        -v /mnt/work:/work \
    connoisseur-image /bin/bash
$ cd /repositories/connoisseur
$ python setup.py install
```

## Running the experiments

After entering the virtual environment or initiating the docker container,
experiments can be found at the `/repositories/connoisseur/experiments`
folder. An execution example follows:

```shell
cd ./experiments/van_gogh/extract-inception-train-svm/
python run.py
```

Each experiment is wrapped by sacred package, responsible for
monitoring the experiment and logging its progression to RECOD's main
mongodb at `mongodb://10.0.1.218:27017/meteor`. However, `python run.py`
will not persist any log. To do so, use the `m` parameter:

```shell
python run.py -m 10.0.1.218:27017:meteor
```
