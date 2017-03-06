"""Extract Patches to Disk.

This experiment consists on the following procedures:

 * Load paintings from the PainterByNumbers dataset.
 * Divide these paintings into patches of an determined size.
 * Save each patch individually onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

from PIL import ImageFile
from sacred import Experiment

from connoisseur import datasets

ex = Experiment('1-extract-patches-to-disk')


@ex.config
def config():
    dataset_seed = 4
    classes = None
    image_shape = [299, 299, 3]
    n_jobs = 6
    data_dir = "/datasets/painter-by-numbers"
    valid_size = .1
    train_n_patches = 10
    test_n_patches = 10
    valid_n_patches = 10
    patches_saving_mode = 'max-gradient'


@ex.automain
def run(dataset_seed, train_n_patches, valid_n_patches, test_n_patches,
        classes, image_shape, data_dir, patches_saving_mode, valid_size, n_jobs):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    (datasets.PainterByNumbers(base_dir=data_dir, image_shape=image_shape,
                               classes=classes, n_jobs=n_jobs,
                               train_n_patches=train_n_patches,
                               valid_n_patches=valid_n_patches,
                               random_state=dataset_seed)
     .prepare()
     .split(fraction=valid_size, phase='valid')
     .save_patches_to_disk(mode=patches_saving_mode))
