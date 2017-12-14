"""PainterByNumbers Dataset.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
import re
import shutil

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder, StandardScaler

from .base import DataSet


def load_multiple_outputs(train_info, outputs_meta, encode='onehot'):
    assert encode in ('onehot', 'sparse'), 'unknown encode %s' % encode

    def to_year(e):
        year_pattern = r'(?:\w*[\s\.])?(\d{3,4})(?:\.0?)?$'
        try:
            return e if isinstance(e, float) else float(re.match(year_pattern, e).group(1))
        except (AttributeError, ValueError):
            print('unknown year', e)
            return np.nan

    y_train = pd.read_csv(train_info, quotechar='"', delimiter=',')

    categorical_output_names = ['artist', 'style', 'genre']
    outputs = {}

    for meta in outputs_meta:
        n = meta['n']
        if n in categorical_output_names:
            en = LabelEncoder()
            is_nan = pd.isnull(y_train[n])
            encoded = en.fit_transform(y_train[n].apply(str).str.lower()).astype('float')
            encoded[is_nan] = np.nan

            flow = make_pipeline(Imputer(strategy='most_frequent'),
                                 OneHotEncoder(sparse=False) if encode == 'onehot' else None)
        else:
            encoded = y_train[n] if n != 'date' else y_train['date'].apply(to_year)
            encoded = encoded.values
            flow = make_pipeline(Imputer(strategy='mean'),
                                 StandardScaler())

        outputs[n] = flow.fit_transform(encoded.reshape(-1, 1))
        meta['f'] = flow

    name_map = {os.path.splitext(n)[0]: i for i, n in enumerate(y_train['filename'])}
    return outputs, name_map


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
