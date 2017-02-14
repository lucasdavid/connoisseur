"""Classify with SVM.

This experiment consists on the following procedures:

 * Load .pickle files and classify them with an sklearn SVM.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ex = Experiment('3-classify-svm')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 256
    image_shape = [299, 299, 3]
    train_shuffle = True
    train_n_patches = 2
    train_augmentations = []
    dataset_train_seed = 12
    valid_shuffle = True
    valid_n_patches = 2
    valid_augmentations = []
    dataset_valid_seed = 98
    test_shuffle = True
    test_n_patches = 80
    dataset_test_seed = 53
    test_augmentations = []
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/paintings91"


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


@ex.automain
def run(data_dir):
    data = {}
    for phase in ('train', 'test'):
        with open(os.path.join(data_dir, 'Paintings91', '%s.pickle' % phase), 'rb') as f:
            data[phase] = pickle.load(f)
            # Data as-it-is and categorical labels.
            data[phase] = data[phase]['data'], np.argmax(data[phase]['target'], axis=-1)

    X, y = data['train']
    X_test, y_test = data['test']

    model = Pipeline([
        ('pca', PCA(n_components=.99)),
        ('svc', SVC(random_state=24))
    ])
    model.fit(X, y)
    p_test = model.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_true=y_test, y_pred=p_test)
    print('# Test\n%s\naccuracy: %.3f'
          % (classification_report(y_true=y_test, y_pred=p_test),
             accuracy_score(y_test, p_test)))

    plot_confusion_matrix(confusion_matrix_test)
