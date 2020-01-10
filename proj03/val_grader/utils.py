"""
Source: https://github.com/philkr/dl_class/blob/master/homework/_03/solution/utils.py
"""


import numpy as np


def _one_hot(x, n):
    return (x.reshape(-1,1) == np.arange(n, dtype=x.dtype)).astype(int)

class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = np.arange(self.size)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds+1, self.size), _one_hot(labels+1, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(axis=0)

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = np.zeros((size, size))
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """

        self.matrix += self._make(preds, labels).astype(float)

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / self.matrix.sum()

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / self.matrix.sum(1)

    @property
    def average_accuracy(self):
        return np.nanmean(self.class_accuracy)
