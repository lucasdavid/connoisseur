"""Extract Patches to Disk.

This experiment consists on the following procedures:

 * Load paintings from the vanGogh dataset.
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
    image_shape = [256, 256, 3]
    n_jobs = 2
    data_dir = "/datasets/van_gogh"
    valid_size = .1
    train_n_patches = 50
    valid_n_patches = 50
    test_n_patches = 50
    pool_size = 4
    patches_saving_mode = 'max-gradient'
    device = '/cpu:0'


@ex.automain
def run(dataset_seed, classes, image_shape, data_dir,
        train_n_patches, valid_n_patches, test_n_patches,
        patches_saving_mode, valid_size, n_jobs, pool_size, device):
    (datasets.VanGogh(base_dir=data_dir,
                      image_shape=image_shape,
                      classes=classes,
                      train_n_patches=train_n_patches,
                      valid_n_patches=valid_n_patches,
                      test_n_patches=test_n_patches,
                      n_jobs=n_jobs,
                      random_state=dataset_seed)
     .prepare()
     .download()
     .extract()
     .split(fraction=valid_size, phase='valid')
     .save_patches_to_disk(mode=patches_saving_mode,
                           pool_size=pool_size,
                           device=device))
