#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin
Inspired from seq2seq code
Dataset prepares/serves batches from the given input data

"""

import constants
from torch.autograd import Variable
import torch

class Dataset(object):

    def __init__(self, srcData, lblData, batchSize, cuda, ft=False, volatile=False):
        """

        :param srcData: Source data, word, subwords and predicate flag (binary mark)
        :param lblData: Gold Labels
        :param batchSize:
        :param cuda: binary var. True if cuda will be used
        :param volatile: Should be True during evaluation (no history will be kept to save space)
        """
        self.wrd = srcData[0]
        self.subs = srcData[1]
        self.bm = srcData[2]
        self.predwrd = srcData[3]

        # use fasttext
        self.useft = ft

        if lblData:
            self.role = lblData[0]
            assert(len(self.wrd) == len(self.role))
        else:
            self.role = None

        self.cuda = cuda
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.otype = torch.cuda.LongTensor if cuda else torch.LongTensor
        self.batchSize = batchSize
        self.numBatches = len(self.wrd) // batchSize
        self.volatile = volatile

    def _batchify_all(self, data):
        return(self._batchify_word(data[0]),
               self._batchify_vec(data[1]),
               self._batchify(data[2]),
               self._batchify(data[3]))

    def _batchify(self, data, padding=constants.PAD):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(padding)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        # move input tensors to gpu if available (or create there)
        if self.cuda:
            out = out.cuda()
        # require_grad by default false
        v = Variable(out, volatile=self.volatile)
        return v

    def _batchify_word(self, data, padding=constants.PAD):
        if not self.useft:
            self._batchify(data)
        else:
            max_length = max(len(x) for x in data)
            padding = u'<UNK>'
            for i, sent in enumerate(data):
                if(len(sent)<max_length):
                    for j in range(len(sent),max_length):
                        data[i].append(padding)
            return data

    def _batchify_vec(self, data,  padding=constants.PAD):
        # return empty if the unit is word (data is not a vector)
        if len(data[0][0].shape)==0:
            return []
        # max sentence length (number of words)
        max_length = max(x.size(0) for x in data)
        # e.g number of characters
        max_sub_unit_size = len(data[0][0])
        out = data[0].new(len(data), max_length, max_sub_unit_size).fill_(padding)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data[i])

        out = out.contiguous()

        if self.cuda:
            out = out.cuda()
        # require_grad by default false
        v = Variable(out, volatile=self.volatile)
        return v


    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch = self._batchify_all([
            self.wrd[index*self.batchSize:(index+1)*self.batchSize],
            self.subs[index*self.batchSize:(index+1)*self.batchSize],
            self.bm[index*self.batchSize:(index+1)*self.batchSize],
            self.predwrd[index*self.batchSize:(index+1)*self.batchSize]])

        if self.role:
            roleBatch = self._batchify(
                self.role[index*self.batchSize:(index+1)*self.batchSize])
        else:
            roleBatch = None

        return srcBatch, roleBatch

    def __len__(self):
        return self.numBatches
