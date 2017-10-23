from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
from keras import backend as K
from keras.preprocessing import image as keras_image
from six.moves import range

try:
    from PIL import Image as pil_image, ImageEnhance
except ImportError:
    pil_image = None


def load_img(path, grayscale=False, target_size=None,
             extraction_method='resize'):
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
                img = img.resize((target_size[1], target_size[0]),
                                 pil_image.ANTIALIAS)

            elif extraction_method == 'central-crop':
                start = np.array([img.width - target_size[1],
                                  img.height - target_size[0]]) / 2
                end = start + (target_size[1], target_size[0])
                img = img.crop((start[0], start[1], end[0], end[1]))

            elif extraction_method == 'random-crop':
                start = (np.random.rand(2) * (img.width - target_size[1],
                                              img.height - target_size[
                                                  0])).astype('int')
                end = start + (target_size[1], target_size[0])
                img = img.crop((start[0], start[1], end[0], end[1]))
    return img


class DirectoryIterator(keras_image.DirectoryIterator):
    def __init__(self, *args, extraction_method='resize', enhancer=None,
                 **kwargs):
        super(DirectoryIterator, self).__init__(*args, **kwargs)
        self.extraction_method = extraction_method
        self.enhancer = enhancer or PaintingEnhancer(augmentations=())

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        if all(self.image_shape):
            batch_x = np.zeros((current_batch_size,) + self.image_shape,
                               dtype=K.floatx())
        else:
            batch_x = np.zeros(current_batch_size, dtype=object)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
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
                img = keras_image.array_to_img(batch_x[i], self.data_format,
                                               scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
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
            batch_y = np.zeros((len(batch_x), self.num_class),
                               dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class PairsDirectoryIterator(DirectoryIterator):
    """Iterator capable of creating pairs of images.

    :param batch_size: size of the batch yielded each next(self) call.
    :param window_size: size of the image window loaded from the disk.
    """

    def __init__(self, *args, batch_size=32, window_size=128, **kwargs):
        super().__init__(*args, batch_size=window_size, **kwargs)
        self.real_batch_size = batch_size
        self.pix = 0
        self.pairs_window = []
        self.pairs_labels_window = []
        self.x_window = None

    def next(self):
        if self.x_window is None:
            x_window, y_window = super().next()

            if self.class_mode == 'categorical':
                y_window = np.argmax(y_window, axis=-1)

            labels, c = np.unique(y_window, return_counts=True)
            indices = [np.where(y_window == l)[0] for l in labels]
            count = c.max()

            pairs = []
            pairs_labels = []
            for i, _u in enumerate(labels):
                for n in range(count):
                    ul = len(indices[_u])
                    pairs += [[indices[_u][n % ul], indices[_u][(n + np.random.randint(1, ul)) % ul]]]

                    dn = (i + np.random.randint(1, len(labels))) % len(labels)
                    dn = labels[dn]
                    dnl = len(indices[dn])
                    pairs += [[indices[_u][n % ul], indices[dn][n % dnl]]]
                    pairs_labels += [1, 0]

            self.x_window = x_window
            self.pairs_window = np.array(pairs)
            self.pairs_labels_window = np.array(pairs_labels)
            self.pix = 0

        c = self.pairs_window[self.pix:self.pix + self.real_batch_size]
        pairs_x = self.x_window[c.T]
        pairs_y = self.pairs_labels_window[self.pix:self.pix + self.real_batch_size]
        self.pix += self.real_batch_size
        if self.pix > len(self.pairs_window):
            self.x_window = None

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
