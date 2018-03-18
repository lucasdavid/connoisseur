"""Extract Patches to Disk.

This experiment consists on the following procedures:

 * Load paintings from the a data set.
 * Divide these paintings into patches of an determined size.
 * Save each patch individually onto the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

from sacred import Experiment

ex = Experiment('extract-patches-to-disk')


@ex.config
def config():
    dataset_seed = 4
    classes = None
    image_shape = [224, 224, 3]
    n_jobs = 1
    dataset_name = 'VanGogh'
    data_dir = '/datasets/vangogh-test-recaptures/recaptures-vangogh-museum/original'
    saving_directory = '/datasets/vangogh-test-recaptures/recaptures-vangogh-museum/original/patches/random_224'
    valid_size = 0
    train_n_patches = 50
    valid_n_patches = 50
    test_n_patches = 50
    downloading = False
    extracting = False
    preparing = False
    pool_size = 4
    patches_saving_mode = 'all'
    device = '/cpu:0'


@ex.automain
def run(dataset_name, dataset_seed, classes, image_shape, data_dir, saving_directory,
        downloading, extracting, preparing, train_n_patches, valid_n_patches, test_n_patches,
        patches_saving_mode, valid_size, n_jobs, pool_size, device):
    from PIL import Image, ImageFile
    import tensorflow as tf
    from connoisseur import datasets

    Image.MAX_IMAGE_PIXELS = None
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
        dataset.save_patches_to_disk(directory=saving_directory, mode=patches_saving_mode, pool_size=pool_size)

    print('done')
