import os
import shutil

import matplotlib
import numpy as np
from keras.preprocessing.image import load_img

matplotlib.use('agg')

from sacred import Experiment

ex = Experiment('convert-cross-dataset')


@ex.config
def my_config():
    data_dir = '/datasets/vangogh/vgdb_2016/valid/'
    output_dir = '/datasets/vangogh/vgdb_2016/valid-small/'
    max_sizes = [1024, 1024]


@ex.automain
def main(data_dir, output_dir, max_sizes):
    print('loading data...')
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for c in os.listdir(data_dir):
        os.makedirs(output_dir + c, exist_ok=True)

        for s in os.listdir(data_dir + c):
            if s.startswith('original'):
                continue

            image = load_img(data_dir + c + '/' + s)
            size = np.array(image.size)
            ratio = max_sizes[0] / size[0]
            image = image.resize((int(size[0] * ratio), int(size[1] * ratio)))
            print(s, size, '-->', image.size)
            image.save(output_dir + c + '/' + s)
