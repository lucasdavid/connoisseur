"""4 Evaluate.

Evaluate predictions made by one or many classifiers.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2017 (c)

"""
import os

from sacred import Experiment

ex = Experiment('van-gogh/4-evaluate')


@ex.config
def config():
    predictions = '/mnt/files/Dropbox/masters-ldavid/van-gogh/predictions'


def evaluate(data, phase):
    import numpy as np
    from sklearn import metrics
    from connoisseur.fusion import strategies

    # from connoisseur.fusion import strategies

    local_data = list(map(lambda x: list(filter(lambda y: y['phase'] == phase, x))[0], data))

    f = []

    for d in local_data:
        samples = np.array(d['samples'])
        p = np.argsort(samples)
        f.append(samples[p])
        d['samples'] = samples[p]

    np.equal(f[0], f[1])
    np.equal(f[1], f[2])

    labels = local_data['labels']
    evaluations = local_data['evaluations']

    print('##', phase, 'phase')

    for strategy_tag in ('raw', 'sum', 'mean', 'farthest', 'most_frequent'):
        print('### strategy_tag', strategy_tag)

        local_evaluations = list(map(lambda x: list(filter(lambda y: y['strategy_tag'] == strategy_tag, x))[0], evaluations))

        print('\n'.join(
            'clf #%i\'s score: %f' % (clf_id, e['score'])
            for clf_id, e in enumerate(local_evaluations)), '\n')

        p = np.concatenate(list(map(lambda x: np.array(x['p']).reshape((-1, 1)), local_evaluations)), -1)
        p = p.T

        r = strategies.most_frequent(p, None, multi_class=False)


# for strategy_tag in ('sum', 'mean', 'farthest', 'most_frequent'):
#         strategy = getattr(strategies, strategy_tag)
#
#         p = SkLearnFusion(model, strategy=strategy).predict(x)
#         score = metrics.accuracy_score(y, p)
#         print('score using', strategy_tag, 'strategy:', score, '\n',
#               metrics.classification_report(y, p),
#               '\nConfusion matrix:\n',
#               metrics.confusion_matrix(y, p))
#         print('samples incorrectly classified:', names[p != y])
#
#         results['evaluations'].append({
#             'strategy': strategy_tag,
#             'score': score,
#             'p': p.tolist()
#         })
#
#     return results


@ex.automain
def run(predictions):
    import json

    print('loading predictions...', end=' ')
    data = []
    for p in os.listdir(predictions):
        with open(os.path.join(predictions, p)) as fp:
            data.append(json.load(fp))
    print('done.\n')

    for d in data:
        d['evauations'][0]

    print('# evaluation report')
    for phase in ('train', 'test'):
        evaluate(data, phase)
