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
    data_dir = '/datasets/vangogh-test-recaptures/vangogh-museum/original/vgdb_2016/test/'
    originals_dir = '/datasets/vangogh/vgdb_2016/'
    output_dir = '/datasets/vangogh-test-recaptures/vangogh-museum/resized/vgdb_2016/test/'
    resize = True


@ex.automain
def main(data_dir, originals_dir, output_dir, resize):
    print('loading data...')
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    labels = os.listdir(os.path.join(originals_dir, 'train'))
    all_sizes = {
        phase: list(map(lambda p: {
            'name': p,
            'path': os.path.join(originals_dir, phase, c, p),
            'label': c,
            'sizes': load_img(os.path.join(originals_dir, phase, c, p)).size
        }, os.listdir(os.path.join(originals_dir, phase, c)))) for c in labels
        for phase in ['train', 'valid', 'test']
        if os.path.exists(originals_dir + phase)
    }

    test_painting_sizes = {os.path.splitext(s['name'])[0]: s['sizes'] for s in all_sizes['test']}
    train_painting_sizes = {os.path.splitext(s['name'])[0]: s['sizes'] for s in all_sizes['train']}

    sizes_ = np.asarray(list(train_painting_sizes.values()))
    horizontals = np.argmax(sizes_, axis=1) == 0
    avg_h_sizes, h_std = sizes_[horizontals].mean(axis=0), sizes_[horizontals].std(axis=0)
    avg_v_sizes, v_std = sizes_[~horizontals].mean(axis=0), sizes_[~horizontals].std(axis=0)

    print('average/std train sizes in %s:' % originals_dir,
          '  all: %s/%s' % (sizes_.mean(axis=0), sizes_.std(axis=0)),
          '  verticals: %s/%s' % (avg_v_sizes, v_std),
          '  horizontals: %s/%s' % (avg_h_sizes, h_std),
          sep='\n', end='\n\n')

    for c in os.listdir(data_dir):
        os.makedirs(output_dir + c, exist_ok=True)

        for s in os.listdir(data_dir + c):
            if s.startswith('original'):
                continue

            original_name = '-'.join(s.split('-')[:-1])

            if resize:
                image = load_img(data_dir + c + '/' + s)
                size = np.array(image.size)
                is_horizontal = size[0] > size[1]
                size_ix = 0 if is_horizontal else 1

                size_ = (test_painting_sizes[original_name]
                         if original_name in test_painting_sizes
                         else avg_h_sizes if is_horizontal
                         else avg_v_sizes)
                std_ = (h_std if is_horizontal else v_std)

                if abs(size - size_)[size_ix] > std_[size_ix]:
                    # Only reshape if difference is above std.
                    ratio = size_[size_ix] / size[size_ix]
                    image = image.resize((int(size[0] * ratio), int(size[1] * ratio)))
                    print(s, size, '-->', image.size)
                else:
                    print(s, size)

                image.save(output_dir + c + '/' + s)
            else:
                shutil.copy(data_dir + c + '/' + s, output_dir + c + '/' + s)
