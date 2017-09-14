from __future__ import absolute_import
from __future__ import print_function

import os
import math

import numpy as np
from keras import backend as K
from keras.preprocessing import image as keras_image
from six.moves import range

try:
    from PIL import Image as pil_image, ImageEnhance
except ImportError:
    pil_image = None


def load_img(path, grayscale=False, target_size=None, extraction_method='resize'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        extraction_method: str, (default='resize')
            {'resize', 'central-crop', 'random-crop'}

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple and any(hw_tuple):
            if extraction_method == 'resize':
                img = img.resize((target_size[1], target_size[0]), pil_image.ANTIALIAS)

            elif extraction_method == 'central-crop':
                start = np.array([img.width - target_size[1], img.height - target_size[0]]) / 2
                end = start + (target_size[1], target_size[0])
                img = img.crop((start[0], start[1], end[0], end[1]))

            elif extraction_method == 'random-crop':
                start = (np.random.rand(2) * (img.width - target_size[1], img.height - target_size[0])).astype('int')
                end = start + (target_size[1], target_size[0])
                img = img.crop((start[0], start[1], end[0], end[1]))
    return img


class DirectoryIterator(keras_image.DirectoryIterator):
    def __init__(self, *args, extraction_method='resize', enhancer=None, **kwargs):
        self.extraction_method = extraction_method
        self.enhancer = enhancer or PaintingEnhancer(augmentations=())
        super(DirectoryIterator, self).__init__(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        if all(self.image_shape):
            batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        else:
            batch_x = np.zeros(current_batch_size, dtype=object)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname), grayscale=grayscale,
                           target_size=self.target_size,
                           extraction_method=self.extraction_method)
            img = self.enhancer.process(img)
            x = keras_image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = keras_image.array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class PairsDirectoryIterator(DirectoryIterator):
    def __init__(self, *args,
                #  anchor_label=1, balance_labels=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.anchor_label = anchor_label
        # self.balance_labels = balance_labels

    def next(self):
        batch_x, batch_y = super().next()
        batch_size = batch_x.shape[0]

        if self.class_mode == 'categorical':
            batch_y = np.argmax(batch_y, axis=-1)

        # samples0, = np.where(batch_y != self.anchor_label)
        # samples1, = np.where(batch_y == self.anchor_label)

        # if not self.balance_labels or not len(samples0) or not len(samples1):
            # Random pairs combination.
        c = np.random.randint(0, batch_size, size=(2, batch_size))
        # else:
            # c0 = np.hstack((
            #     np.random.choice(samples1, size=(math.floor(batch_size / 2), 1)),
            #     np.random.choice(samples0, size=(math.floor(batch_size / 2), 1))))
            # c1 = np.random.choice(samples1, size=(math.ceil(batch_size / 2), 2))
            # c = np.vstack((c0, c1)).T

        pairs_x = batch_x[c]
        pairs_y = batch_y[c]
        pairs_y = (pairs_y[0] == pairs_y[1]).astype(np.float)

        return list(pairs_x), pairs_y


class PairsNumpyArrayIterator(keras_image.NumpyArrayIterator):
    def next(self):
        batch_x, batch_y = super().next()
        batch_size = batch_x.shape[0]

        if len(batch_y.shape) > 1:
            batch_y = np.argmax(batch_y, axis=-1)

        c = np.random.randint(0, batch_size, size=(2, batch_size))

        pairs_x = batch_x[c]
        pairs_y = batch_y[c]
        pairs_y = (pairs_y[0] == pairs_y[1]).astype(np.float)

        return list(pairs_x), pairs_y


class PaintingEnhancer:
    def __init__(self, augmentations=('color', 'brightness', 'contrast'),
                 variability=0.25):
        self.augmentations = augmentations
        self.variability = variability

    def process(self, patch):
        if 'color' in self.augmentations:
            enhance = ImageEnhance.Color(patch)
            patch = enhance.enhance(self.variability * np.random.randn() + 1)

        if 'brightness' in self.augmentations:
            enhance = ImageEnhance.Brightness(patch)
            patch = enhance.enhance(self.variability * np.random.randn() + 1)

        if 'contrast' in self.augmentations:
            enhance = ImageEnhance.Contrast(patch)
            patch = enhance.enhance(self.variability * np.random.randn() + 1)
        return patch
