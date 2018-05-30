#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin
Inspired from seq2seq - OpenNMT code
"""

import torch.optim as optim
from torch.nn.utils import clip_grad_norm

class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, lr_decay=1, patience=3):
        self.last_f1 = 0
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.patience = patience
        self.start_decay = False
        self.minrun = patience+1

    def step(self):
        """
        Compute gradients norm.
        :return:
        """
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self, f1, epoch):
        """
        Decay learning rate if val perf does not improve or we hit the start_decay_at limit
        The number of iterations allowed before decaying the learning rate if there is no improvement on dev set
        :param f1:
        :param epoch:
        :return:
        """
        self.start_decay = False
        diff = f1 - self.last_f1
        if diff < 0.01:
            self.start_decay = True
        # If no improvement and it has run for minimum required epoch
        # decrease patience
        if self.patience is not None and \
                epoch >= self.minrun and \
                         self.start_decay:
            self.patience-=1
        # If have no patience for this thing
        if self.patience==0 and self.start_decay:
            self.lr = self.lr * self.lr_decay
            # reset patience for the new learning rate
            self.patience = self.minrun-1
            # reset decay to false
            self.start_decay = False
            print("Decaying learning rate to %g" % self.lr)
        # register new values
        self.last_f1 = f1
        self.optimizer.param_groups[0]['lr'] = self.lr

