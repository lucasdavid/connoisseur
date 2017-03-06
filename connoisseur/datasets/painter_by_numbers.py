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

        for phase in ('train',):
            file_name = os.path.join(base_dir, '%s_info.csv' % phase)
            phase_path = os.path.join(base_dir, phase)

            frame = pd.read_csv(file_name, quotechar='"', delimiter=',')
            self.feature_names_ = frame.columns.values

            # Organize files in a Keras-friendly representation.
            data = frame.values

            for painting_name, artist_url in zip(data[:, 0], data[:, 1]):
                os.makedirs(os.path.join(phase_path, str(artist_url)), exist_ok=True)

                try:
                    shutil.move(
                        os.path.join(phase_path, painting_name),
                        os.path.join(phase_path, str(artist_url), painting_name)
                    )
                except FileNotFoundError:
                    # Already moved. Ignore.
                    pass

        return self
