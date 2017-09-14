"""4 Evaluate With A Meta-Classifier.

Evaluate predictions made by one or many classifiers by training a classifier on top of it.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2017 (c)

"""
import os

import numpy as np
from sacred import Experiment
from sklearn import metrics

ex = Experiment('vangogh/4-evaluate-metaclassifier')


@ex.config
def config():
    predictions = '/mnt/files/Dropbox/masters-ldavid/vangogh/predictions'


def filter_dataset(data, phase):
    data = list(map(lambda x: list(filter(lambda y: y['phase'] == phase, x['phases']))[0], data))

    for d in data:
        p = np.argsort(d['samples'])
        d['samples'] = np.array(d['samples'])[p]
        d['labels'] = np.array(d['labels'])[p]

        for e in d['evaluations']:
            e['p'] = np.array(e['p'])

            if e['strategy'] == 'raw':
                e['p'] = e['p'].reshape(p.shape[0], d['patches_count'])

            e['p'] = e['p'][p]
    return data


def train(data):
    data = filter_dataset(data, 'train')
    samples = data[0]['samples']
    labels = data[0]['labels']
    evaluations = list(map(lambda x: x['evaluations'], data))

    return model


def evaluate(meta_model, data, phase):
    from connoisseur.fusion import strategies

    print('estimators:', list(map(lambda x: x['estimator'], data)))
    data = filter_dataset(data, phase)

    samples = data[0]['samples']
    labels = data[0]['labels']
    evaluations = list(map(lambda x: x['evaluations'], data))

    print('##', phase, 'phase')

    for strategy_tag in ('sum', 'mean', 'farthest', 'most_frequent'):
        print('### strategy_tag', strategy_tag)

        local_evaluations = list(map(lambda x: list(filter(lambda y: y['strategy'] == strategy_tag, x))[0],
                                     evaluations))

        print('\n'.join(
            'clf #%i\'s score: %f (missing: %s)' % (clf_id, e['score'], samples[e['p'] != labels])
            for clf_id, e in enumerate(local_evaluations)))

        p = np.concatenate(list(map(lambda x: x['p'].reshape(1, *x['p'].shape), local_evaluations)), 0).T
        p = strategies.most_frequent(p, None, multi_class=False)

        print('combining with most_frequent strategy:\n',
              metrics.classification_report(labels, p),
              '\nConfusion matrix:\n',
              metrics.confusion_matrix(labels, p), '\n',
              'samples incorrectly classified:', samples[p != labels], '\n')


@ex.automain
def run(predictions):
    import json

    print('loading predictions...', end=' ')
    data = []
    for p in os.listdir(predictions):
        with open(os.path.join(predictions, p)) as fp:
            data.append({'estimator': p, 'phases': json.load(fp)})
    print('done.\n')

    meta_model = train(data)

    print('# evaluation report')
    for phase in ('train', 'test'):
        evaluate(meta_model, data, phase)
