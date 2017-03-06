import time

import matplotlib.pyplot as plt
from sacred import Experiment

from connoisseur import datasets

ex = Experiment('tests-with-gradients')


@ex.config
def config():
    dataset = '/datasets/van_gogh_test/'
    results = './output-max-gradient-reduced-images'
    patch_size = [299, 299]
    n_patches = 4
    low_threshold = .9
    device = '/cpu:0'
    mode = 'max-gradient'


def save(images, name):
    for image_id, image in enumerate(images):
        ax = plt.subplot(1, len(images), image_id + 1)
        ax.axis("off")
        ax.imshow(image)

    plt.tight_layout()
    plt.savefig(name)
    plt.clf()


@ex.automain
def run(dataset, results, patch_size, n_patches, mode, low_threshold, device):
    start = time.time()
    (datasets.VanGogh(base_dir=dataset, image_shape=patch_size,
                      train_n_patches=n_patches,
                      valid_n_patches=n_patches,
                      test_n_patches=n_patches)
     .prepare()
     .download()
     .extract()
     .save_patches_to_disk(mode=mode, device=device, low_threshold=low_threshold))

    print('done (%.2f s)' % (time.time() - start))
