import csv
import os

import numpy as np
from scipy.stats import itemfreq
from sklearn.preprocessing import LabelEncoder


class Wikiart:
    """WikiArt Data Set.

    Notes:
        'batch_size': 50,
        'height': 600,
        'width': 600,
        'channels': 3,

        'train_validation_test_split': [.8, .2],

        'n_classes': 'all',
        'save_in': '/tmp',
        'n_epochs': None,
        'check_images': False
    """

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.label_encoder = None

    def load(self):
        params = self.parameters
        filename = os.path.join(params['save_in'], 'wikiart.data')

        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')

            for _ in range(13):
                # Eliminate header.
                next(reader)

            image_names, target = zip(*reader)

        images_folder = os.path.join(params['save_in'], 'images')
        image_names = list(map(
            lambda f: os.path.join(images_folder, f + '.jpg'),
            image_names))

        image_names, target = map(np.array, (image_names, target))

        if params['check_images']:
            for image in image_names:
                # Let's just make sure all files are in place
                # before any hard computation.
                assert os.path.exists(image)

        freq = dict(itemfreq(target))
        permutation = sorted(range(target.shape[0]),
                             key=lambda x: freq[target[x]],
                             reverse=True)
        image_names = image_names[permutation]
        target = target[permutation]

        n_classes = params['n_classes']
        if n_classes != 'all':
            cut_point, unique_count, last = 1, 0, target[0]
            while unique_count < n_classes and cut_point < target.shape[0]:
                if target[cut_point] != last:
                    unique_count += 1
                    last = target[cut_point]
                cut_point += 1
            cut_point -= 1

            image_names, target = image_names[:cut_point], target[:cut_point]

        # Transform painters' codes into integers.
        self.label_encoder = LabelEncoder()
        target = self.label_encoder.fit_transform(target)

        self._build_all_input_pipelines(image_names, target)
        return self
