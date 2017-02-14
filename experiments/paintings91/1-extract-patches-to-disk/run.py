import sacred

from connoisseur.datasets import Paintings91

ex = sacred.Experiment('1-extract-patches-to-disk')


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = '/datasets/ldavid/paintings91'


@ex.automain
def run(data_dir, image_shape):
    (Paintings91(base_dir=data_dir, image_shape=image_shape)
     .download()
     .extract()
     .split_train_test()
     .extract_patches_to_disk())
