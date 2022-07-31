import pandas as pd
import numpy as np
from hdf5storage import loadmat
from contextlib import contextmanager
from functools import wraps
import time


def time_this(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


@contextmanager
def time_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} running time: {}s'.format(label, end - start))


def load_data(dataset_name: str, path='./'):
    try:
        if dataset_name == 'http':
            print('Dataset: {}'.format(dataset_name))
            mat = loadmat(f'{path}http.mat')
            X, y = mat['X'], mat['y'].astype(np.int64)
            print(f'Dataset Shape\nX: {X.shape}\ny: {y.shape}')
            return X, y.reshape(-1)

        first_word = dataset_name.split('-')[0]
        if first_word in ['Friday', 'Wednesday', 'Thursday']:
            print('Dataset: {}-2018_processed'.format(dataset_name))
            filename = f'{path}{dataset_name}-2018_processed.csv'
            data = pd.read_csv(filename)
            y = data['label'].values
            X = data.drop(['id', 'label'], axis=1)
            print(f'Dataset Shape\nX: {X.shape}\ny: {y.shape}')
            return X, y.reshape(-1)

    except Exception as e:
        print(e)


