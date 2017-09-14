"""Extract Patches to Disk.

This experiment consists on the following procedures:

 * Load paintings from the a data set.
 * Divide these paintings into patches of an determined size.
 * Save each patch individually onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

from sacred import Experiment

ex = Experiment('1-extract-patches-to-disk')


@ex.config
def config():
    dataset_seed = 4
    classes = None
    image_shape = [256, 256, 3]
    n_jobs = 2
    dataset_name = 'VanGogh'
    data_dir = "/home/ldavid/datasets/vangogh"
    valid_size = .1
    train_n_patches = 50
    valid_n_patches = 50
    test_n_patches = 50
    downloading = True
    extracting = True
    preparing = True
    pool_size = 4
    patches_saving_mode = 'random'
    device = '/cpu:0'


@ex.automain
def run(dataset_name, dataset_seed, classes, image_shape, data_dir,
        downloading, extracting, preparing,
        train_n_patches, valid_n_patches, test_n_patches,
        patches_saving_mode, valid_size, n_jobs, pool_size, device):
    from PIL import ImageFile

    import tensorflow as tf
    from connoisseur import datasets

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    dataset_cls = getattr(datasets, dataset_name)
    dataset = dataset_cls(base_dir=data_dir,
                          image_shape=image_shape,
                          classes=classes,
                          train_n_patches=train_n_patches,
                          valid_n_patches=valid_n_patches,
                          test_n_patches=test_n_patches,
                          n_jobs=n_jobs,
                          random_state=dataset_seed)
    if downloading:
        dataset.download()
    if extracting:
        dataset.extract()
    if preparing:
        dataset.prepare()
    if valid_size > 0:
        dataset.split(fraction=valid_size, phase='valid')
    with tf.device(device):
        dataset.save_patches_to_disk(mode=patches_saving_mode, pool_size=pool_size)
    print('done')
