import time

from sacred import Experiment

import matplotlib.pyplot as plt

ex = Experiment('tests-with-gradients')


@ex.config
def config():
    dataset = '/datasets/van_gogh_test/'
    patch_size = [299, 299]
    n_patches = 4
    low_threshold = .9
    device = '/cpu:0'
    mode = 'max-gradient'
    pool_size = 2


def save(images, name):
    for image_id, image in enumerate(images):
        ax = plt.subplot(1, len(images), image_id + 1)
        ax.axis("off")
        ax.imshow(image)

    plt.tight_layout()
    plt.savefig(name)
    plt.clf()


@ex.automain
def run(dataset, patch_size, n_patches, mode, low_threshold, device, pool_size):
    import tensorflow as tf
    from connoisseur import datasets

    start = time.time()

    with tf.device(device):
        (datasets.VanGogh(base_dir=dataset, image_shape=patch_size,
                          train_n_patches=n_patches,
                          valid_n_patches=n_patches,
                          test_n_patches=n_patches)
         .prepare()
         .download()
         .extract()
         .save_patches_to_disk(mode=mode,
                               low_threshold=low_threshold,
                               pool_size=pool_size))

    print('done (%.2f s)' % (time.time() - start))
