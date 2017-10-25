"""PainterByNumbers Dataset.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
import shutil

import pandas as pd

from .base import DataSet


class PainterByNumbers(DataSet):
    def prepare(self, override=False):
        print('preparing PainterByNumbers...')

        base_dir = self.full_data_path

        fn = os.path.join(base_dir, 'train_info.csv')

        frame = pd.read_csv(fn, quotechar='"', delimiter=',')
        self.feature_names_ = frame.columns.values
        data = frame.values

        for sample in data:
            fn, label = sample[:2]
            os.makedirs(os.path.join(base_dir, 'train', label), exist_ok=True)
            try:
                shutil.move(os.path.join(base_dir, 'train', fn),
                            os.path.join(base_dir, 'train', label, fn))
            except FileNotFoundError:
                # Already moved. Ignore.
                pass

        os.makedirs(os.path.join(base_dir, 'test', 'unknown'), exist_ok=True)
        for sample in os.listdir(os.path.join(base_dir, 'test')):
            if os.path.isfile(os.path.join(base_dir, 'test', sample)):
                shutil.move(os.path.join(base_dir, 'test', sample),
                            os.path.join(base_dir, 'test', 'unknown', sample))
        return self
