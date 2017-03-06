"""WikiArt Dataset.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
import shutil

import pandas as pd
from PIL import ImageFile

from .base import DataSet


class WikiArt(DataSet):
    def prepare(self, override=False):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        base_dir = self.full_data_path
        file_name = os.path.join(base_dir, 'wikiart.data')
        images_path = os.path.join(base_dir, 'images')
        paintings_path = os.path.join(base_dir, 'train')

        frame = pd.read_csv(file_name, skiprows=10, quotechar='"', delimiter=',')
        self.feature_names_ = frame.columns.values

        if not os.listdir(images_path):
            # Empty list of files. They were already moved.
            return self

        # Organize files in a Keras-friendly representation.
        data = frame.values

        for painting_id, artist_url in zip(data[:, 0], data[:, 5]):
            os.makedirs(os.path.join(paintings_path, str(artist_url)), exist_ok=True)
            paintings_name = '%i.jpg' % painting_id

            try:
                shutil.move(
                    os.path.join(images_path, paintings_name),
                    os.path.join(paintings_path, str(artist_url), paintings_name)
                )
            except FileNotFoundError:
                # Already moved. Ignore.
                pass

        return self
