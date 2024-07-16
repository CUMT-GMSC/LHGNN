import torch
import numpy as np

class HEBatchGenerator(object):
    def __init__(self, hyperedges, labels, batch_size, device):
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.shuffle()

    def __iter__(self):
        self._cursor = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return self.next_batch()

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]

    def next_batch(self):
        ncursor = self._cursor + self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:]
            labels = self.labels[self._cursor:]
            self._cursor = 0
            hyperedges = [torch.LongTensor(list(edge)).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            self.shuffle()
            return hyperedges, labels, True

        hyperedges = self.hyperedges[self._cursor:self._cursor + self.batch_size]

        labels = self.labels[self._cursor:self._cursor + self.batch_size]

        hyperedges = [torch.LongTensor(list(edge)).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)

        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False