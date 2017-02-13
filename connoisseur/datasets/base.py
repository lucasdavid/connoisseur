"""Connoisseur DataSet Base Class.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import itertools
import os
import shutil
import tarfile
import zipfile
from urllib import request

import numpy as np
from PIL import ImageEnhance
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from ..utils.image import img_to_array, load_img


class PaintingEnhancer(object):
    def __init__(self, augmentations=('color', 'brightness', 'contrast'), variability=0.5):
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


class DataSet(object):
    """DataSet Base Class.

    Parameters
    ----------
    base_dir: the directory where the dataset should be downloaded and
        extracted.

    load_mode: ('exact', 'balanced')
        Mode in which the samples are loaded. Options are:
        --- exact: `train_n_patches` patches are extracted from every painting
            in the dataset.
        --- balanced: `train_n_patches` patches are extracted from every
            painting in the dataset. Finally, some paintings are discarded
            until .

    train_n_patches: int, default=50, quantity of patches to extract from
        every training painting.
    test_n_patches: int, default=50, quantity of patches to extract from
        every test painting.

    train_augmentations: list
        List for allowed augmentations performed during loading. Valid values
        are 'color', 'brightness' and 'contrast'.
    test_augmentations: list
        Similar to `train_augmentations`, but applied to `test_data`.

    min_label_rate: float \in [0, 1].
        Minimum rate allowed for labels. All samples of a label which violates
        this bound will be removed before stored in `train_data` and
        `test_data`.
    """

    SOURCE = None
    COMPACTED_FILE = 'dataset.zip'
    EXPECTED_SIZE = 0
    DATA_SUB_DIR = None

    generator = train_data = valid_data = test_data = None

    def __init__(self, base_dir='./data', load_mode='exact',
                 train_n_patches=50, valid_n_patches=50, test_n_patches=50,
                 image_shape=(224, 224, 3),
                 classes=None, min_label_rate=0,
                 train_augmentations=(),
                 valid_augmentations=(),
                 test_augmentations=(),
                 valid_split=None,
                 random_state=None):
        if not isinstance(valid_split, (int, float)):
            ValueError('valid_split not understood: %s' % valid_split)

        self.load_mode = load_mode
        self.base_dir = base_dir
        self.train_n_patches = train_n_patches
        self.valid_n_patches = valid_n_patches
        self.test_n_patches = test_n_patches
        self.image_shape = image_shape
        self.classes = classes
        self.min_label_rate = min_label_rate
        self.train_augmentations = train_augmentations
        self.train_enhancer = PaintingEnhancer(train_augmentations)
        self.valid_augmentations = valid_augmentations
        self.valid_enhancer = PaintingEnhancer(valid_augmentations)
        self.test_augmentations = test_augmentations
        self.test_enhancer = PaintingEnhancer(test_augmentations)
        self.valid_split = valid_split
        self.random_state = check_random_state(random_state)

        self.label_encoder_ = None

    @property
    def full_data_path(self):
        return (os.path.join(self.base_dir, self.DATA_SUB_DIR)
                if self.DATA_SUB_DIR
                else self.base_dir)

    def download(self, override=False):
        os.makedirs(self.base_dir, exist_ok=True)
        file_name = os.path.join(self.base_dir, self.COMPACTED_FILE)

        if os.path.exists(file_name):
            stat = os.stat(file_name)
            if stat.st_size == self.EXPECTED_SIZE and not override:
                print(self.COMPACTED_FILE, 'download skipped.')
                return self

            print('copy corrupted. Re-downloading dataset.')

        print('downloading', self.SOURCE)
        file_name, _ = request.urlretrieve(self.SOURCE, file_name)
        stat = os.stat(file_name)
        print('%s downloaded (%i bytes).' % (self.COMPACTED_FILE, stat.st_size))

        if self.EXPECTED_SIZE and stat.st_size != self.EXPECTED_SIZE:
            raise RuntimeError('File does not have expected size: (%i/%i)' % (stat.st_size, self.EXPECTED_SIZE))
        return self

    def extract(self, override=False):
        zipped = os.path.join(self.base_dir, self.COMPACTED_FILE)

        if len(os.listdir(self.base_dir)) > 1 and not override:
            print(self.COMPACTED_FILE, 'extraction skipped.')
        else:
            print('extracting', zipped)
            extractor = self._get_specific_extractor(zipped)
            extractor.extractall(self.base_dir)
            extractor.close()

            print('dataset extracted.')
        return self

    @staticmethod
    def _get_specific_extractor(zipped):
        ext = os.path.splitext(zipped)[1]

        if ext in ('.tar', '.gz', '.tar.gz'):
            return tarfile.open(zipped)
        elif ext == '.zip':
            return zipfile.ZipFile(zipped, 'r')
        else:
            raise RuntimeError('Cannot extract %s. Unknown format.' % zipped)

    def check(self):
        assert os.path.exists(self.full_data_path), 'Data set not found. Have you downloaded and extracted it first?'

        base = self.full_data_path

        if self.valid_split and not os.path.exists(os.path.join(base, 'valid')):
            print('splitting train and valid data...')
            labels = os.listdir(os.path.join(base, 'train'))
            files = [list(map(lambda x: os.path.join(l, x),
                              os.listdir(os.path.join(base, 'train', l))))
                     for l in labels]
            files = np.array(list(itertools.chain(*files)))
            self.random_state.shuffle(files)

            valid_split = (self.valid_split
                           if isinstance(self.valid_split, int)
                           else int(files.shape[0] * self.valid_split))

            print('%i/%i files will be used for validation.' % (valid_split, files.shape[0]))
            train_files, valid_files = files[valid_split:], files[:valid_split]

            os.mkdir(os.path.join(base, 'valid'))
            for l in labels:
                os.mkdir(os.path.join(base, 'valid', l))

            for file in valid_files:
                shutil.move(os.path.join(base, 'train', file), os.path.join(base, 'valid', file))
        print('checked.')
        return self

    def load_patches_from_full_images(self, *phases):
        print('loading %s images' % ','.join(phases))
        phases = phases or ('train', 'valid', 'test')

        data_path = self.full_data_path
        image_shape = self.image_shape
        labels = self.classes or os.listdir(os.path.join(data_path, 'train'))

        n_samples_per_label = np.array([len(os.listdir(os.path.join(data_path, 'train', label))) for label in labels])
        rates = n_samples_per_label / n_samples_per_label.sum()

        if 'train' in phases:
            print('labels\'s rates: %s' % dict(zip(labels, np.round(rates, 2))))
            print('min tolerated label rate: %.2f' % self.min_label_rate)

        labels = list(map(lambda i: labels[i],
                          filter(lambda i: rates[i] >= self.min_label_rate,
                                 range(len(labels)))))
        min_n_samples = n_samples_per_label.min()

        for phase in phases:
            X, y = [], []

            n_patches = getattr(self, '%s_n_patches' % phase)
            enhancer = getattr(self, '%s_enhancer' % phase)

            if not n_patches:
                continue

            print('extracting %i %s patches...' % (n_patches, phase))
            for label in labels:
                class_path = os.path.join(data_path, phase, label)

                samples = os.listdir(class_path)

                if phase == 'train' and self.load_mode == 'balanced':
                    self.random_state.shuffle(samples)
                    samples = samples[:min_n_samples]

                for name in samples:
                    full_name = os.path.join(class_path, name)
                    img = load_img(full_name)

                    patches = []

                    for _ in range(n_patches):
                        start = (self.random_state.rand(2) *
                                 (img.width - image_shape[1],
                                  img.height - image_shape[0])).astype('int')
                        end = start + (image_shape[1], image_shape[0])
                        patch = img.crop((start[0], start[1], end[0], end[1]))

                        patch = enhancer.process(patch)
                        patches.append(img_to_array(patch))

                    X.append(patches)
                    y.append(label)

            print('%s patches extraction to memory completed.' % phase)

            X = np.array(X, dtype=np.float)
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)

            setattr(self, '%s_data' % phase, (X, y))
        print('loading completed.')
        return self

    def unload(self, *phases):
        phases = phases or ('train', 'valid', 'test')

        for phase in phases:
            setattr(self, '%s_data' % phase, None)

        return self

    def get(self, phase):
        return getattr(self, '%s_data' % phase)

    def extract_patches_to_disk(self):
        print('extracting patches to disk...')

        data_path = self.full_data_path
        sizes = self.image_shape
        labels = self.classes or os.listdir(os.path.join(data_path, 'train'))

        patches_path = os.path.join(data_path, 'extracted_patches')

        os.makedirs(patches_path, exist_ok=True)

        phases = ('train', 'test')
        if os.path.exists(os.path.join(data_path, 'valid')):
            phases += 'valid',

        for phase in phases:
            if os.path.exists(os.path.join(patches_path, phase)):
                print('%s patches extraction to disk skipped.' % phase)
                continue

            print('extracting %s patches to disk...' % phase)

            for label in labels:
                class_path = os.path.join(data_path, phase, label)
                patches_class_path = os.path.join(patches_path, phase, label)
                os.makedirs(patches_class_path)

                for name in os.listdir(class_path):
                    full_name = os.path.join(class_path, name)
                    img = load_img(full_name)

                    n_patches = 0

                    for dx in range(sizes[1], img.width + 1, sizes[1]):
                        for dy in range(sizes[0], img.height + 1, sizes[0]):
                            e = np.array([dx, dy])
                            s = e - (sizes[1], sizes[0])

                            img.crop((s[0], s[1], e[0], e[1])).save(
                                os.path.join(patches_class_path,
                                             '%s-%i-%i.jpg' % (os.path.splitext(name)[0], dx, dy)))
                            n_patches += 1
            print('patches extraction completed.')
        return self
