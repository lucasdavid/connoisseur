import os

from keras.preprocessing.image import load_img, img_to_array
from pandas.io.json import json_normalize
from sacred import Experiment

ex = Experiment('recaptures-physical')


@ex.config
def config():
    originals_dataset = '/datasets/vangogh/vgdb_2016/test'
    recaptures_dataset = '/datasets/vangogh-test-recaptures/recaptures-google/original/vgdb_2016/test'
    metrics = ['width', 'height', 'r', 'g', 'b']
    misses = ['vg_9103139-2', 'vg_9414279-0', 'vg_9378884-3', 'vg_9413420-1',
              'vg_9421984-0', 'vg_9414279-2', 'vg_9103139-3', 'vg_9386980-0',
              'vg_9110201-1', 'vg_9106795-1', 'vg_9387502-1', 'nvg_10500055-0',
              'nvg_10582548-2', 'vg_9103139-0', 'vg_9103139-1', 'vg_9387502-3',
              'vg_9414279-1', 'vg_9463012-0', 'vg_9386980-1', 'nvg_9780042-2',
              'vg_9506505-1']


@ex.automain
def main(originals_dataset, recaptures_dataset, metrics, misses):
    print('originals:', originals_dataset)

    originals, recaptures = [], []
    for d, o in ((originals_dataset, originals), (recaptures_dataset, recaptures)):
        for c in os.listdir(d):
            for f in os.listdir(os.path.join(d, c)):
                i = load_img(os.path.join(d, c, f))
                a = img_to_array(i)
                o += [{'name': os.path.splitext(f)[0],
                       'width': i.size[0],
                       'height': i.size[1],
                       **dict(zip(('r', 'g', 'b'),
                                  a.mean(axis=(0, 1))))}]

    originals = json_normalize(originals)
    originals.set_index('name', inplace=True)
    recaptures = json_normalize(recaptures)
    recaptures.set_index('name', inplace=True)
    recaptures['original'] = [i.split('-')[0] for i in recaptures.index]

    print('average values:')
    recaptures_avg = recaptures[metrics]
    print(recaptures_avg.groupby(recaptures.index.isin(misses)).mean())

    r = recaptures.join(originals, on='original', rsuffix='_o')

    for m in metrics:
        r[m + '_std'] = (r[m] - r[m + '_o']) ** 2

    print('standard deviations:')
    recaptures_var = r[[c for c in r.columns if c.endswith('_std')]]
    print(recaptures_var.groupby(r.index.isin(misses)).mean() ** (1 / 2))
