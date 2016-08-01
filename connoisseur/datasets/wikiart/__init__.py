import os
import csv
import numpy as np

from .. import base


class Wikiart(base.ImageDataSet):
    NAME = 'wikiart'

    DEFAULT_PARAMETERS = {}

    def load(self):
        params = self.parameters

        data_file = os.path.join(params['save_in'], 'wikiart.data')
        print('loading %s' % data_file)
        convert = lambda x: float(x.strip() or -999)
        data = np.genfromtxt(data_file, delimiter=',', converters=convert)

        print('data')
        print(data[:1000])

        return self
