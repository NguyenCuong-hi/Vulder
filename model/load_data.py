import pickle

import numpy as np
from imblearn.over_sampling import RandomOverSampler


def load_dataset():
    X_test = pickle.load(open("D:/FreeLanceProject/Data/X_test.npy", "rb"))
    y_test = pickle.load(open("D:/FreeLanceProject/Data/y_test.npy", "rb"))
    X_train = pickle.load(open("D:/FreeLanceProject/Data/X_train.npy", "rb"))
    y_train = pickle.load(open("D:/FreeLanceProject/Data/y_train.npy", "rb"))

    y_train = y_train.argmax(axis=1)
    y_test = y_test.argmax(axis=1)


    return X_test, y_test, X_train, y_train
    pass


def oversampling(X_train, y_train):
    over = RandomOverSampler(sampling_strategy=0.5)
    index = np.arange(0, len(y_train)).reshape(-1, 1)
    x_index, y = over.fit_resample(index, y_train)
    index = x_index.reshape(1, -1)[0]
    print(y_train[index].shape)

    X_train = [X_train[i][index] for i in range(0, 3)]
    y_train = y_train[index]
    return X_train, y_train
    pass

if __name__ == '__main__':
    load_dataset()
