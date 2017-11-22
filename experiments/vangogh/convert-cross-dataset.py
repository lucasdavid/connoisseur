import os
import shutil

import matplotlib
import numpy as np
from keras.preprocessing.image import load_img

matplotlib.use('agg')

from sacred import Experiment

ex = Experiment('check-dataset')


@ex.config
def my_config():
    data_dir = '/datasets/vangogh-test-recaptures/recaptures/'
    originals_dir = '/datasets/vangogh/vgdb_2016/test/'
    output_dir = '/datasets/vangogh-test-recaptures/recaptures/'
    resize = False


@ex.automain
def main(data_dir, originals_dir, output_dir, resize):
    print('loading data...')
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    labels = os.listdir(originals_dir)
    samples = [list(map(lambda p: {
        'name': p,
        'path': originals_dir + c + '/' + p,
        'label': c,
        'sizes': load_img(originals_dir + c + '/' + p).size
    }, os.listdir(originals_dir + c))) for c in labels]
    samples = np.concatenate(samples)
    sizes = {os.path.splitext(s['name'])[0]: s['sizes'] for s in samples}

    for c in os.listdir(data_dir):
        os.makedirs(output_dir + c, exist_ok=True)

        for s in os.listdir(data_dir + c):
            for i in os.listdir(data_dir + c + '/' + s):
                if i.startswith('original'):
                    continue

                if resize:
                    image = load_img(data_dir + c + '/' + s + '/' + i)
                    original_size = sizes[s]
                    size = image.size
                    ratio = original_size[0] / size[0]
                    image = image.resize((int(size[0] * ratio), int(size[1] * ratio)))
                    print(original_size, '-->', image.size)
                    image.save(output_dir + c + '/' + s + '-' + i)
                else:
                    shutil.copy(data_dir + c + '/' + s + '/' + i,
                                output_dir + c + '/' + s + '-' + i)
