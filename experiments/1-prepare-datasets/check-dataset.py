import os

import numpy as np
from keras.preprocessing.image import load_img
from sacred import Experiment

ex = Experiment('check-dataset')


@ex.config
def my_config():
    data_dir = '/datasets/vangogh/vgdb_2016/test/'


@ex.automain
def main(data_dir):
    print('loading data...')
    labels = os.listdir(data_dir)

    samples = [list(map(lambda p: {
        'path': data_dir + c + '/' + p,
        'label': c,
        'sizes': load_img(data_dir + c + '/' + p).size
    }, os.listdir(data_dir + c))) for c in labels]
    samples = np.concatenate(samples)

    print('number of samples:', len(samples))
    print(samples[:10], '...')

    sizes = np.asarray([s['sizes'] for s in samples])

    print('min sizes:', sizes.min(axis=0))
    print('avg sizes:', sizes.mean(axis=0))
    print('max sizes:', sizes.max(axis=0))

    areas = np.prod(sizes, axis=1)
    print('min  area:', areas.min(axis=0))
    print('avg  area:', areas.mean(axis=0))
    print('max  area:', areas.max(axis=0))
