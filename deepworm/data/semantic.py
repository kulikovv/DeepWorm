import random
from os import listdir
from os.path import join, isfile

import numpy as np
from skimage.io import imread


class SemanticWorms:
    def __init__(self, path, balance=False):
        self.path = path

        files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        self.all_names = [n[:-len('.tif')] for n in files if n.endswith('.tif')]

        # split data
        self.all_names = np.random.permutation(self.all_names)
        number = len(self.all_names)
        self.train_names = self.all_names[:-number / 10 * 2]
        self.test_names = self.all_names[-number / 10 * 2:-number / 10]
        self.valid_names = self.all_names[-number / 10:]

        self.cache = {}

        self.train_probs = [1. / float(len(self.train_names))] * len(self.train_names)
        self.balance = balance
        if balance:
            self.train_probs = self.build_stats()

    def build_stats(self):

        leaf_stats = []
        for j in range(len(self.train_names)):
            x, y, m = self.get_train(j)
            leafs = []
            for i in range(1, y.shape[0]):
                leaf = np.sum(y[i])
                if leaf > 0:
                    leafs.append(leaf)
            leaf_stats.append(np.mean(leafs))

        hist = np.histogram(np.array(leaf_stats), bins=5)
        leaf_prob = np.array([1.] * len(leaf_stats))
        for i in range(len(hist[0])):
            pos = np.where(np.logical_and(leaf_stats >= hist[1][i], leaf_stats <= hist[1][i + 1]))
            leaf_prob[pos] = 1. - hist[0][i] / float(len(leaf_stats))
        return leaf_prob / np.sum(leaf_prob)

    def make_all_train(self):
        self.train_names = self.all_names
        self.train_probs = [1. / float(len(self.train_names))] * len(self.train_names)
        if self.balance:
            self.train_probs = self.build_stats()

    def __get_image__(self, basename):
        rgb = imread(join(self.path, basename + '.tif')).astype(np.float32)
        label = imread(join(self.path, basename + '_labels.png'), 'pil')
        # label[label>0]=1
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) - 0.5

        return rgb, label

    def __get_labeled__(self, names, n=None):
        if n is None:
            if 1 == len(names):
                n = 0
            else:
                n = np.random.randint(0, len(names) - 1)

        name = names[n]
        if name in self.cache:
            return self.cache[name]

        # print(name)
        rgb, binary = self.__get_image__(name)

        self.cache[name] = (rgb, binary)
        return self.cache[name]

    def get_test(self, n=None):
        x, y = self.__get_image__(self.test_names[n])
        return x, y, None

    def get_train(self, n=None):
        if n is None:
            n = np.random.choice(range(len(self.train_names)), 1, p=self.train_probs)[0]
        return self.__get_labeled__(self.train_names, n)

    def get_valid(self, n=None):
        return self.__get_labeled__(self.valid_names, n)

    def __get_batch__(self, get_func, batch_size=3, transforms=[]):
        images = []
        labels = []

        for t in transforms:
            if hasattr(t, 'prepare'):
                t.prepare()

        for index in range(batch_size):
            i, l = get_func(None)

            i = i
            l = l

            for t in transforms:
                seed = random.randint(0, 2 ** 32 - 1)
                np.random.seed(seed)
                i = t(i, True)
                np.random.seed(seed)
                l = t(l)

            images.append(np.expand_dims(np.expand_dims(i, 0), 0))
            labels.append(np.expand_dims(l, 0))

        labels = np.clip(np.vstack(labels).astype(np.int), 0, 3)

        return np.vstack(images), labels

    def get_train_batch(self, batch_size=3, transforms=[]):
        return self.__get_batch__(self.get_train, batch_size=batch_size, transforms=transforms)

    def get_valid_batch(self, batch_size=3, transforms=[]):
        return self.__get_batch__(self.get_valid, batch_size=batch_size, transforms=transforms)
