"""Connoisseur DataSet Base Class.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import itertools
import os
import pickle
import shutil
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from urllib import request

import numpy as np
import tensorflow as tf
from PIL import ImageOps
from keras.preprocessing.image import img_to_array, load_img
from skimage import feature
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from ..utils.image import PaintingEnhancer


def load_pickle_data(data_dir, phases=None, keys=None, chunks=(0,),
                     layers=None, classes=None):
    phases = phases or ('train', 'valid', 'test')
    keys = keys or ('data', 'target', 'names')

    data = {p: {k: [] for k in keys} for p in phases}

    for p in phases:
        for c in chunks:
            file_name = os.path.join(data_dir, '%s.%i.pickle' % (p, c))

            if os.path.exists(file_name):
                with open(file_name, 'rb') as file:
                    d = pickle.load(file)
                    for k in keys:
                        data[p][k].append(d[k])
            elif c > 0:
                print('skipping', file_name)
            else:
                raise ValueError('%s data cannot be found at %s.' % (p, data_dir))

        _layers = layers or data[p]['data'][0].keys()

        for k in keys:
            if k == 'data':
                # Merges chunks of each layer output.
                data[p][k] = {l: np.concatenate([x[l] for x in data[p][k]]) for l in _layers}
            else:
                # Merges chunks integrally.
                data[p][k] = np.concatenate(data[p][k])

        if classes is not None:
            if isinstance(classes, int):
                classes = range(classes)
            classes = list(classes)

            s = np.in1d(data[p]['target'], classes)
            for k in keys:
                if k == 'data':
                    for l in data[p][k].keys():
                        data[p][k][l] = data[p][k][l][s]
                else:
                    data[p][k] = data[p][k][s]

        data[p] = tuple(data[p][k] for k in keys)
    return data


def group_by_paintings(*arrays,
                       names: np.ndarray,
                       max_patches: int = None):
    # Aggregate test patches by their respective paintings.
    _x, _y, _names = [], [], []
    outputs = [[] for _ in arrays]
    # Remove patches indices, leaving just the painting name.
    clipped_names = np.array(['-'.join(n.split('-')[:-1]) for n in names])
    for name in set(clipped_names):
        s = clipped_names == name

        indices, = np.where(s)
        if max_patches and len(indices) > max_patches:
            indices = indices[:max_patches]

        for _i, _o in zip(arrays, outputs):
            _o.append(_i[indices])

        _names.append(clipped_names[indices][0])

    return [np.asarray(o) for o in outputs] + [np.asarray(_names)]


def _load_patch_coroutine(options):
    return img_to_array(
        PaintingEnhancer(options['augmentations'],
                         variability=options.get('variability', .25)).process(
            load_img(options['name'])))


def _load_patches_coroutine(args):
    name, patch_size, n_patches, augmentations, random_state = args
    random_state = check_random_state(random_state)
    img = load_img(name)
    patches = []
    enhancer = PaintingEnhancer(augmentations)
    for _ in range(n_patches):
        start = (random_state.rand(2) *
                 (img.width - patch_size[0],
                  img.height - patch_size[1])).astype('int')
        end = start + patch_size
        patch = img.crop((start[0], start[1], end[0], end[1]))

        patch = enhancer.process(patch)
        patches.append(img_to_array(patch))

    return patches


def _save_image_patches_coroutine(options):
    image = load_img(options['name'])
    border = np.array(options['patch_size']) - image.size
    painting_name = os.path.splitext(os.path.basename(options['name']))[0]
    patches_path = options['patches_path']

    if np.any(border > 0):
        # The image is smaller than the patch size in any dimension.
        # Pad it to make sure we can extract at least one patch.
        border = np.ceil(border.clip(0, border.max()) / 2).astype(np.int)
        image = ImageOps.expand(image, border=tuple(border))

    mode = options['mode']
    patch_size = options.get('patch_size', [256, 256])
    n_patches = options['n_patches']

    if mode in ('min-gradient', 'max-gradient'):
        gray_image = image.convert('L')
        gray_tensor = img_to_array(gray_image).squeeze(-1)
        e = feature.canny(gray_tensor, low_threshold=options['low_threshold'], use_quantiles=True).astype(np.float)

        pool_size = options.get('pool_size', 2)
        x, y = options['tensors']

        p = y.eval(feed_dict={x: e.reshape((1,) + e.shape + (1,))})
        p = p.squeeze((0, -1))

        p = np.exp(p / p.sum())
        p /= p.sum()

        if mode == 'min-gradient':
            p = 1 - p
            p /= p.sum()

        c = np.random.choice(np.arange(np.product(p.shape)), size=(n_patches, 1), p=p.flatten())
        c = np.concatenate((c // p.shape[1], c % p.shape[1]), axis=-1).astype(np.int)
        c += np.array(patch_size) // (2 * pool_size)  # restore sizes before convolution
        c *= pool_size  # restore sizes before max_pooling2d
        c -= np.array(patch_size) // 2  # center selected pixels

        starting_points = np.array([c[:, 1], c[:, 0]]).T

    elif mode == 'random':
        starting_points = (
            np.random.rand(n_patches, 2)
            * (image.width - patch_size[0], image.height - patch_size[1])
        ).astype(np.int)

    elif mode == 'all':
        d_widths = list(range(0, image.width - patch_size[0] + 1, patch_size[0]))
        d_heights = list(range(0, image.height - patch_size[1] + 1, patch_size[1]))
        starting_points = itertools.product(d_widths, d_heights)
    else:
        raise ValueError('unknown mode %s' % mode)

    for patch_id, (s_w, s_h) in enumerate(starting_points):
        (image
            .crop((s_w, s_h, s_w + patch_size[0], s_h + patch_size[1]))
            .save(os.path.join(patches_path, '%s-%i.jpg' % (painting_name, patch_id))))


class DataSet:
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
    EXTRACTED_FOLDER = None

    def __init__(self, base_dir='./data', load_mode='exact',
                 train_n_patches=50, valid_n_patches=50, test_n_patches=50,
                 image_shape=(224, 224, 3),
                 classes=None, min_label_rate=0,
                 train_augmentations=(),
                 valid_augmentations=(),
                 test_augmentations=(),
                 n_jobs=1,
                 random_state=None):
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
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)

        self.label_encoder_ = None
        self.feature_names_ = None

    @property
    def full_data_path(self):
        return (os.path.join(self.base_dir, self.EXTRACTED_FOLDER)
        if self.EXTRACTED_FOLDER
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
        print('%s downloaded (%i bytes).'
              % (self.COMPACTED_FILE, stat.st_size))

        if self.EXPECTED_SIZE and stat.st_size != self.EXPECTED_SIZE:
            raise RuntimeError('File does not have expected size: (%i/%i)'
                               % (stat.st_size, self.EXPECTED_SIZE))
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

    def prepare(self, override=False):
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
        assert os.path.exists(
            self.full_data_path), ('Data set not found. Have you downloaded '
                                   'and extracted it first?')
        return self

    def split(self, fraction, phase='valid'):
        base = self.full_data_path

        if os.path.exists(os.path.join(base, phase)):
            print('train-%s splitting skipped.' % phase)
            return self

        print('splitting train-%s data...' % phase)

        labels = os.listdir(os.path.join(base, 'train'))
        files = [list(map(lambda x: os.path.join(l, x),
                          os.listdir(os.path.join(base, 'train', l))))
                 for l in labels]
        files = np.array(list(itertools.chain(*files)))
        self.random_state.shuffle(files)

        faction = (fraction if isinstance(fraction, int) else
        int(files.shape[0] * fraction))

        print('%i/%i files will be used for %s.' % (
            faction, files.shape[0], phase))
        train_files, phase_values = files[faction:], files[:faction]

        for l in labels:
            os.makedirs(os.path.join(base, phase, l), exist_ok=True)

        for file in phase_values:
            shutil.move(os.path.join(base, 'train', file),
                        os.path.join(base, phase, file))
        print('splitting done.')
        return self

    def load_patches_from_full_images(self, *phases):
        phases = phases or ('train', 'valid', 'test')
        print('loading %s images' % ','.join(phases))

        results = []
        data_path = self.full_data_path
        # Keras (height, width) -> PIL Image (width, height)
        patch_size = self.image_shape
        patch_size = [patch_size[1], patch_size[0]]
        labels = self.classes or os.listdir(os.path.join(data_path, 'train'))

        n_samples_per_label = np.array(
            [len(os.listdir(os.path.join(data_path, 'train', label)))
             for label in labels])
        rates = n_samples_per_label / n_samples_per_label.sum()

        if 'train' in phases:
            print('labels\'s rates:',
                  ', '.join(['%s: %.2f' % (l, r)
                             for l, r in zip(labels, rates)]))
            print('minimum label rate tolerated: %.2f' % self.min_label_rate)

        labels = list(map(lambda i: labels[i],
                          filter(lambda i: rates[i] >= self.min_label_rate,
                                 range(len(labels)))))
        min_n_samples = n_samples_per_label.min()

        for phase in phases:
            n_patches = getattr(self, '%s_n_patches' % phase)
            augmentations = getattr(self, '%s_augmentations' % phase)

            X, y, names = [], [], []

            for label in labels:
                class_path = os.path.join(data_path, phase, label)

                samples = os.listdir(class_path)

                if phase == 'train' and self.load_mode == 'balanced':
                    self.random_state.shuffle(samples)
                    samples = samples[:min_n_samples]

                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    patches_per_sample = list(executor.map(
                        _load_patches_coroutine,
                        [(os.path.join(class_path, n), patch_size, n_patches,
                          augmentations, r)
                         for r, n in enumerate(samples)]))

                X += patches_per_sample
                y += len(samples) * [label]
                names += samples

            if phase == 'train':
                self.label_encoder_ = LabelEncoder().fit(y)

            if self.label_encoder_ is None:
                raise ValueError('you need to load train data first in order '
                                 'to initialize the label encoder that will '
                                 'be used to transform the %s data.' % phase)
            y = self.label_encoder_.transform(y)
            results.append([np.array(X, copy=False), y,
                            np.array(names, copy=False)])
        print('loading completed.')
        return results

    def load_patches(self, *phases):
        phases = phases or ('train', 'valid', 'test')

        results = []
        data_path = self.full_data_path
        labels = self.classes or os.listdir(os.path.join(data_path, 'train'))
        r = self.random_state

        for phase in phases:
            n_patches = getattr(self, '%s_n_patches' % phase)
            enhancer = getattr(self, '%s_enhancer' % phase)

            X, y, names = [], [], []

            for label in labels:
                label_sample_path = os.path.join(data_path, phase, label)
                label_patch_path = os.path.join(data_path,
                                                'extracted_patches',
                                                phase, label)
                samples_names = [os.path.splitext(p)[0]
                                 for p in os.listdir(label_sample_path)]
                patches_names = os.listdir(label_patch_path)

                for sample in samples_names:
                    sample_patches_names = list(filter(lambda x: sample in x,
                                                       patches_names))
                    if (n_patches is not None and
                        len(sample_patches_names) < n_patches):
                        sample_patches_names = r.choice(sample_patches_names,
                                                        n_patches)
                    else:
                        r.shuffle(sample_patches_names)

                    sample_patches_names = sample_patches_names[:n_patches]

                    with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                        _patches = list(executor.map(
                            _load_patch_coroutine,
                            [dict(name=os.path.join(label_patch_path, sample_patch_name),
                                  augmentations=enhancer.augmentations,
                                  variability=enhancer.variability)
                             for sample_patch_name in sample_patches_names]))

                    X.append(_patches)
                    y.append(label)
                    names.append(sample)

            if phase == 'train':
                self.label_encoder_ = LabelEncoder().fit(y)

            if self.label_encoder_ is None:
                raise ValueError('you need to load train data first in order '
                                 'to initialize the label encoder that will '
                                 'be used to transform the %s data.' % phase)
            y = self.label_encoder_.transform(y)
            results.append([np.array(X, copy=False), y,
                            np.array(names, copy=False)])
        return results

    def save_patches_to_disk(self, directory, mode='all', low_threshold=.9, pool_size=2):
        """Extract and save patches to disk.

        :param directory: str, directory in which the patches will be saved.
        :param mode: str, mode in which patches are extracted. Options are:
            * 'all': all available patches are extracted.
            * 'random': random patches are extracted.
            * 'max-gradient':
                extract patches randomly, where the probability of extracting
                a patch p is the normalized intensity of the gradient in the
                pixels contained in the patch.
            * 'min-gradient':
                same as 'max-gradient', but using 1-p as probability
                distribution, where p used is the gradient intensity map.
        :param low_threshold: threshold passed to canny filter when deciding
            upon which pixels definitely do not belong to edges.
            Ignored if mode != 'max-gradient'.
        :param pool_size: max-pool size. Scalar by which the input image is
            reduced before convolved with the kernels of ones. Higher values
            will decrease the accuracy of the procedure but decrease in time
            and memory requirements.
            Ignored if mode != 'max-gradient'.

        :return: self
        """
        print('saving patches to disk...')

        data_path = self.full_data_path
        patch_size = self.image_shape
        # Keras (height, width) -> PIL Image (width, height)
        patch_size = [patch_size[1], patch_size[0]]

        os.makedirs(directory, exist_ok=True)

        phases = list(filter(lambda x: os.path.exists(os.path.join(data_path, x)),
                             ('train', 'test', 'valid')))

        tensors = None

        if mode in ('min-gradient', 'max-gradient'):
            with tf.name_scope('max_gradient_patches'):
                x = tf.placeholder(tf.float32, shape=(1, None, None, 1))

                with tf.name_scope('max_pool_1'):
                    y = tf.layers.average_pooling2d(x, pool_size=pool_size, strides=pool_size)

                with tf.name_scope('conv_1'):
                    kernel_weights = tf.ones([patch_size[0] // pool_size, patch_size[1] // pool_size, 1, 1],
                                             name='kernel')
                    y = tf.nn.conv2d(y, kernel_weights, strides=(1, 1, 1, 1), padding="VALID", name='op')

            tensors = (x, y)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config):
            tf.global_variables_initializer().run()

            for phase in phases:
                n_patches = getattr(self, '%s_n_patches' % phase)

                print('extracting %s patches to disk...' % phase)

                labels = self.classes or os.listdir(os.path.join(data_path, phase))
                for label in labels:
                    class_path = os.path.join(data_path, phase, label)
                    patches_label_path = os.path.join(directory, phase, label)
                    os.makedirs(patches_label_path, exist_ok=True)

                    samples = os.listdir(class_path)

                    for sample in samples:
                        try:
                            if os.path.exists(os.path.join(patches_label_path, os.path.splitext(sample)[0] + '-0.jpg')):
                                continue

                            _save_image_patches_coroutine(dict(
                                name=os.path.join(class_path, sample),
                                patches_path=patches_label_path,
                                patch_size=patch_size,
                                n_patches=n_patches,
                                mode=mode,
                                low_threshold=low_threshold,
                                pool_size=pool_size,
                                tensors=tensors))
                        except MemoryError:
                            print('failed')

        print('patches extraction completed.')
        return self
