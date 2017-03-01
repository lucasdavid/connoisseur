"""Extract Patches to Disk.

This experiment consists on the following procedures:

 * Load paintings from the vanGogh dataset.
 * Divide these paintings into patches of an determined size.
 * Save each patch individually onto the disk.

Author: Lucas David -- <ld492@drexel.edu>
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
    data_dir = "/datasets/ldavid/van_gogh"
    valid_split = .1


@ex.automain
def run(dataset_seed, classes, image_shape, data_dir, valid_split, n_jobs):
    (datasets.VanGogh(base_dir=data_dir, image_shape=image_shape,
                      classes=classes, n_jobs=n_jobs,
                      random_state=dataset_seed)
     .download()
     .extract()
     .split_train_valid(valid_split)
     .extract_patches_to_disk())
