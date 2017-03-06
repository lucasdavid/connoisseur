"""Extract Patches to Disk.

This experiment consists on the following procedures:

 * Load paintings from the WikiArt dataset.
 * Divide these paintings into patches of an determined size.
 * Save each patch individually onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

from sacred import Experiment

from connoisseur import datasets

ex = Experiment('1-extract-patches-to-disk')


@ex.config
def config():
    dataset_seed = 4
    classes = None
    image_shape = [299, 299, 3]
    n_jobs = 8
    data_dir = "/datasets/wikiart"
    valid_size = .1
    test_size = .2


@ex.automain
def run(dataset_seed, classes, image_shape, data_dir, valid_size, test_size, n_jobs):
    (datasets.WikiArt(base_dir=data_dir, image_shape=image_shape,
                      classes=classes, n_jobs=n_jobs,
                      random_state=dataset_seed)
     .prepare()
     .split(fraction=test_size, phase='test')
     .split(fraction=valid_size, phase='valid')
     .save_patches_to_disk())
