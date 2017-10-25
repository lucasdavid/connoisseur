from __future__ import absolute_import
from __future__ import print_function

import os
import math

import numpy as np
from keras import backend as K
from keras.preprocessing import image as ki
from six.moves import range
from keras.utils.data_utils import Sequence

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


class DirectoryIterator(ki.DirectoryIterator):
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
            x = ki.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = ki.array_to_img(batch_x[i], self.data_format,
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


class MultipleOutputsSequence(Sequence):
    """Iterator capable of creating (images, {painters, styles, ...}).

    :param batch_size: size of the batch yielded each next(self) call.
    """

    def __init__(self, directory, y, image_data_generator, batch_size=32,
                 target_size=None, classes=None):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.target_size = target_size
        self.classes = np.asarray(classes or sorted(os.listdir(directory)))

        if not directory.endswith('/'):
            directory += '/'

        _id = 0
        samples = {}
        classes = []
        for c in self.classes:
            files = os.listdir(directory + c)
            if files:
                samples[_id] = list(map(lambda _f: directory + c + '/' + _f, files))
                classes.append(c)
                _id += 1
        self.classes = np.asarray(classes)

        x, y = [], []

        for c1 in range(len(self.classes)):
            x += np.random.choice(samples[c1], pairs).reshape(int(pairs / 2), 2).tolist()
            y += int(pairs / 2) * [1.0]

            others = (np.random.randint(1, len(self.classes), size=int(pairs / 2)) + c1) % len(self.classes)
            x += zip(np.random.choice(samples[c1], int(pairs / 2)), (np.random.choice(samples[c2]) for c2 in others))
            y += int(pairs / 2) * [0.0]

        p = np.arange(len(x))
        np.random.shuffle(p)
        self.x, self.y = [np.asarray(e)[p] for e in (x, y)]

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [[], []]
        # build batch of image data
        for a, b in batch_files:
            for i, n in enumerate((a, b)):
                x = ki.img_to_array(ki.load_img(n, target_size=self.target_size))
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] += [x]

        batch_x = [np.asarray(_x) for _x in batch_x]
        return batch_x, np.array(batch_y)


class BalancedDirectoryPairsSequence(Sequence):
    """Iterator capable of creating pairs of images.

    :param batch_size: size of the batch yielded each next(self) call.
    """

    def __init__(self, directory, image_data_generator, batch_size=32,
                 pairs=50, target_size=None, classes=None):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.target_size = target_size
        self.classes = np.asarray(classes or sorted(os.listdir(directory)))

        if not directory.endswith('/'):
            directory += '/'

        _id = 0
        samples = {}
        classes = []
        for c in self.classes:
            files = os.listdir(directory + c)
            if files:
                samples[_id] = list(map(lambda _f: directory + c + '/' + _f, files))
                classes.append(c)
                _id += 1
        self.classes = np.asarray(classes)

        x, y = [], []

        for c1 in range(len(self.classes)):
            x += np.random.choice(samples[c1], pairs).reshape(int(pairs / 2), 2).tolist()
            y += int(pairs / 2) * [1.0]

            others = (np.random.randint(1, len(self.classes), size=int(pairs / 2)) + c1) % len(self.classes)
            x += zip(np.random.choice(samples[c1], int(pairs / 2)), (np.random.choice(samples[c2]) for c2 in others))
            y += int(pairs / 2) * [0.0]

        p = np.arange(len(x))
        np.random.shuffle(p)
        self.x, self.y = [np.asarray(e)[p] for e in (x, y)]

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [[], []]
        # build batch of image data
        for a, b in batch_files:
            for i, n in enumerate((a, b)):
                x = ki.img_to_array(ki.load_img(n, target_size=self.target_size))
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] += [x]

        batch_x = [np.asarray(_x) for _x in batch_x]
        return batch_x, np.array(batch_y)


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
