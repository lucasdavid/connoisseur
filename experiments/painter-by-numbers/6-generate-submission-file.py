"""Generate Submission File.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json

from sacred import Experiment

ex = Experiment('generate-submission-file')


@ex.config
def config():
    results_file_name = '/work/painter-by-numbers/predictions/results-pbn_random_299_inception_auc.json'
    submission_file = './answer.csv'


@ex.automain
def run(results_file_name, submission_file):
    with open(results_file_name) as file:
        data = json.load(file)

    p = data[0]['evaluations'][0]['binary_probabilities']
    print(p[:10])

    with open(submission_file, 'w') as f:
        f.write('index,sameArtist\n')
        f.writelines(['%i,%f\n' % (i, _p) for i, _p in enumerate(p)])
