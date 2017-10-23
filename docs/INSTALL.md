## Connoisseur Installation

### Straight Installation

The simplest way to install connoisseur is the following command:

```shell
$ cd /path/to/connoisseur
$ python3 setup.py install --user
```

As you are installing the package in a global environment, some
incompatibilities might happen.
Try [virtualenv](https://virtualenv.pypa.io/) if that's the case.

### Docker

This method is preferable, as I attached a Dockerfile in this repository.
An image build from the Dockerfile will spawn containers that already have
the correct dependencies to run the experiments.

To use it, build the image:

```shell
$ cd /path/to/connoisseur/docs/
$ docker build -t connoisseur-image .
```

Then run the container and install connoisseur:

```shell
$ nvidia-docker run -it --rm -v /path/to/connoisseur:/connoisseur \
    connoisseur-image /bin/bash
$ cd /connoisseur
$ python setup.py install
```
