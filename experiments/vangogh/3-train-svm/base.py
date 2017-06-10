import os
import pickle

from sklearn.model_selection import train_test_split


def load_data(data_dir, phases=None, share_val_samples=None,
              random_state=None):
    data = {}
    for p in phases or ('train', 'valid', 'test'):
        file_name = os.path.join(data_dir, '%s.pickle' % p)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                d = pickle.load(file)
                data[p] = d['data'], d['target'], d['names']

    if 'valid' not in data and share_val_samples:
        # Separate train and valid sets.
        X, y, names = data['train']
        (X_train, X_valid,
         y_train, y_valid,
         names_train, names_valid) = train_test_split(
            X, y, names, test_size=share_val_samples,
            random_state=random_state)

        data['train'] = X_train, y_train, names_train
        data['valid'] = X_valid, y_valid, names_valid

    return data
