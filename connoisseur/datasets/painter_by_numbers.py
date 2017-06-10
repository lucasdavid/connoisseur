"""PainterByNumbers Dataset.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
import shutil
import unicodedata
from logging import warning

import pandas as pd

from .base import DataSet


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class PainterByNumbers(DataSet):
    def prepare(self, override=False):
        print('preparing PainterByNumbers...')

        warning('up to this point, PainterByNumbers#prepare() only prepares train data')

        base_dir = self.full_data_path

        file_name = os.path.join(base_dir, 'train_info.csv')

        frame = pd.read_csv(file_name, quotechar='"', delimiter=',')
        self.feature_names_ = frame.columns.values
        data = frame.values

        for painting in data:
            file_name, artist = painting[:2]

            phase = 'train'
            src = os.path.join(base_dir, phase, file_name)
            os.makedirs(os.path.join(base_dir, phase, artist), exist_ok=True)

            dst = os.path.join(base_dir, phase, artist, file_name)

            try:
                shutil.move(src, dst)
            except FileNotFoundError:
                # Already moved. Ignore.
                warning('%s not found' % src)

        return self
