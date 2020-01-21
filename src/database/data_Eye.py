
import os
import glob
import numpy as np
import os.path as osp
import csv


class _ExerciseOculo(object):
    """
    """

    def __init__(self, fname, signal, annotations=None, infos=None):
        self.fname = osp.basename(fname)
        self.signal = np.array(signal).astype(float)
        self.annotations = annotations
        self.infos = infos


def load_eyefant(fname):
    X = []
    delimiter = '\t'
    full_fname = fname
    with open(full_fname, encoding="ISO-8859-1") as f:
        nb_sig = int(f.readline().replace('Number of signals:', ''))
        for i in range(5):
            f.readline()
        f.readline()
        info = []
        for i in range(nb_sig):
            info.append(f.readline().strip().split(delimiter)[1:])
        f.readline()
        for line in f.readlines():
            line = line.strip().replace(',', '.')
            X.append(line.split(delimiter)[1:])

    X = np.array(X)
    try:
        if X.shape[1] == 7:
            X = X[::8, :-1]
    except ValueError:
        pass

    if X.shape[1] == 6:
        X = X[:, 2:]


    return _ExerciseOculo(full_fname, X)


