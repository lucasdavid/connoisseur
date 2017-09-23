import os

from sacred import Experiment

ex = Experiment('avg-sample-area1')


@ex.config
def config():
    data_dir = '/datasets/vangogh/vgdb_2016/'
    phases = ('train', 'valid', 'test')


@ex.automain
def main(data_dir, phases):
    import numpy as np
    from keras.preprocessing.image import load_img

    for phase in phases:
        sizes = []
        labels = os.listdir(os.path.join(data_dir, phase))

        try:
            for label in labels:
                samples = os.listdir(os.path.join(data_dir, phase, label))

                for sample in samples:
                    image = load_img(os.path.join(data_dir, phase, label, sample))
                    sizes.append(image.size)

        except KeyboardInterrupt:
            pass

        sizes = np.array(sizes)

        print('average', phase, 'sizes:', sizes.mean(axis=0))
        print('average', phase, 'area:', sizes.prod(axis=1).mean())
