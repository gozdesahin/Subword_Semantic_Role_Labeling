#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Stack Generalization Model

"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit


class SGModelSimple(nn.Module):
    def __init__(self, num_models, num_labels, opt):
        super(SGModelSimple, self).__init__()
        self.num_models = num_models
        self.hiddim = opt.hiddim
        self.outdim = num_labels
        self.ensembler = nn.Linear(num_models,self.hiddim)
        self.sig = F.sigmoid
        self.mapper = nn.Linear(self.hiddim,1)
        self.init_weights(opt.param_init_type, opt.init_scale)

    def init_weights(self, init_type, init_scale):
        # Initialize weight matrix
        for p in self.parameters():
            if init_type == "orthogonal" and p.dim() >= 2:
                nninit.orthogonal(p)
            elif init_type == "uniform":
                p.data.uniform_(-init_scale, init_scale)
            elif init_type == "xavier_n" and p.dim() >= 2:
                nninit.xavier_normal(p)
            elif init_type == "xavier_u" and p.dim() >= 2:
                nninit.xavier_uniform(p)

    def forward(self, input):
        """
        :param input: num_models x seq x num_labels
        :return: label scores
        """
        # convert input to seq x num_labels x num_models
        input = input.permute(1, 2, 0).contiguous()
        nonlin = self.sig(self.ensembler(input.view(-1,self.num_models)))
        label_space = self.mapper(nonlin).view(-1,self.outdim)
        label_scores = F.log_softmax(label_space)
        return label_scores