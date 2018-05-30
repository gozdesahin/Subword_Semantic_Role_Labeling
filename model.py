#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin

Main SRL Model
SRL: Performs SRL given word embeddings from the Subword Encoder
BiLSTMModel - Subword Encoder: Generates word embeddings from the subword units
"""

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit

from IO.util import *

class SRL(nn.Module):
    """
        Encoder: f(word embedding from subword model)
        Input: [predicate binary mask|Encoder output|pretrained word embedding (if any)]
        srl_lstm(input)-> hidden2tag(hidden)-> softmax(labels) -> label probabilities
    """
    def __init__(self, args, ems=None):
        super(SRL, self).__init__()

        # Calculate the input feature size with given subword unit
        # and pretrained word embedding information
        if (args.pre_word_vecs != None) and args.unit == "word":
            args.word_vec_size = self.word_dim

        self.inputFeatSize = self.get_input_dim(args)

        # SRL LSTM and hidden2tag parameters
        self.hidden_dim = args.hidden_size
        self.numLayers = args.layers
        self.numDir = args.numdir
        self.role_size = args.role_size
        self.wp = args.wp
        self.word_dim = args.word_dim

        # if you want to use pretrained word vectors
        if (args.pre_word_vecs != None) or args.unit == "word":
            self.wordEmbeddings = nn.Embedding(args.vocab_size, args.word_vec_size, padding_idx=0)
            self.word_vec_size = args.word_vec_size
            if args.fixed_embed:
                self.wordEmbeddings.weight.requires_grad = False
        else:
            self.wordEmbeddings = None

        ### General parameters
        self.batch_size = args.batch_size
        self.dtype = args.dtype
        self.otype = args.otype
        self.use_cuda = args.use_cuda
        self.options = args
        self.dropout = nn.Dropout(args.dropout)

        ### SRL LSTM
        self.srlLSTM = nn.LSTM(self.inputFeatSize, \
                                     self.hidden_dim, \
                                     num_layers=self.numLayers, \
                                     bidirectional=(self.numDir == 2), \
                                     batch_first=True, \
                                     dropout=args.dropout)

        # go to label space
        self.hidden2tag = nn.Linear(self.hidden_dim*self.numDir, self.role_size)
        # Initialize weights and forget gates
        self.init_weights(args.param_init_type, args.init_scale)
        self.init_forget_gates(value=0.)

        # Initialize subword models
        if args.unit == "word":
            # Module List for subword models
            self.subwordModel = None
        else:
            self.subwordModel = BiLSTMModel(args)
            if args.use_cuda:
                self.subwordModel = self.subwordModel.cuda()

        # initialize word embeddings
        if ems != None:
            # init embeddings from pretrained vectors
            self.wordEmbeddings.weight.data.copy_(torch.FloatTensor(ems).type(dtype))

    def init_hidden(self, numlayer, numdirec=1, hidsize=200, batchsize=32):
        result = (autograd.Variable(torch.zeros(numlayer * numdirec, batchsize, hidsize).type(self.dtype)),
                  autograd.Variable(torch.zeros(numlayer * numdirec, batchsize, hidsize).type(self.dtype)))
        return result

    # Calculate the dimension of input to SRL LSTM / or output of word encoder
    def get_input_dim(self, args):
        # 1 for predicate flag for each case
        inputFeatSize = 1
        # word encoding output will be in word dimension
        if args.composition in ["bi-lstm", "add-bi-lstm"]:
            inputFeatSize += args.word_dim
        # add pretrained word embedding size
        if args.pre_word_vecs:
            inputFeatSize += args.word_vec_size
        elif args.unit == "word":
            inputFeatSize += args.word_vec_size
        return inputFeatSize

    def get_input_repr(self, batch):
        word_ind = batch[0]
        pred_mark_feat = batch[2]
        sequence_length = pred_mark_feat.size(1)
        predmark_feats = pred_mark_feat.view(self.batch_size, sequence_length, 1)

        # initialize input feat with predmark feats
        inputfeat = predmark_feats
        if self.subwordModel != None:
            word_embeds_from_sub = self.subwordModel(batch[1])
            inputfeat = torch.cat((word_embeds_from_sub, inputfeat), 2)

        if (self.wordEmbeddings != None):
            pretr_embeds = self.wordEmbeddings(word_ind)
            inputfeat = torch.cat((pretr_embeds, inputfeat), 2)

        return inputfeat

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
        # Initialize bias for the linear layer
        self.hidden2tag.bias.data.fill_(0.0)

    # In 2014 Zarembi paper it is initialized to 0, but
    # TF tutorial says 1 may give better results ?
    def init_forget_gates(self, value=1.):
        for names in self.srlLSTM._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.srlLSTM, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(value)

    def forward(self, batch):
        # input feat is calculated by submodules
        inputfeat = self.get_input_repr(batch)
        # Initialize SRL hidden state with zeros
        self.srl_hidden = self.init_hidden(self.numLayers, numdirec=self.numDir, hidsize=self.hidden_dim, batchsize=self.batch_size)
        lstm_out, self.srl_hidden = self.srlLSTM(inputfeat, self.srl_hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out = self.dropout(lstm_out)

        # hidden2tag
        tag_space = self.hidden2tag(lstm_out.view(-1, lstm_out.size(2)))
        tag_scores = F.log_softmax(tag_space)

        return tag_scores

class BiLSTMModel(nn.Module):

    def __init__(self, args):
        super(BiLSTMModel, self).__init__()

        ### General parameters
        self.batch_size = args.batch_size
        self.dtype = args.dtype
        self.otype = args.otype
        self.use_cuda = args.use_cuda
        self.options = args

        #### Subword parameters
        self.unit = args.unit
        self.sub_num_layers = args.sub_num_layers
        self.sub_rnn_size = args.sub_rnn_size
        self.word_dim = args.word_dim
        subword_vocab_size = args.subword_vocab_size

        ### Dynamic parameters
        if args.unit == 'char':
            subword_dim = args.char_dim
        else:
            subword_dim = args.morph_dim
        self.subword_dim = subword_dim

        ####### C2W
        self.comp_lstm = nn.LSTM(subword_dim, \
                               self.sub_rnn_size, \
                               num_layers=self.sub_num_layers,\
                               bidirectional=True, \
                               batch_first=True,\
                               dropout=args.dropout
                            )

        # word embedding is calculated as
        # w_t = W_f.h_forward + W_b.h_backward + b
        self.W_f = nn.Parameter(torch.randn(self.sub_rnn_size, self.word_dim))
        self.W_b = nn.Parameter(torch.randn(self.sub_rnn_size, self.word_dim))
        self.we_bias = nn.Parameter(torch.randn(self.word_dim))

        # Initialize weights uniformly
        self.init_weights(args.param_init_type, args.init_scale)
        # subword embedding lookup
        self.subEmbeddings = nn.Embedding(subword_vocab_size, subword_dim, padding_idx=0)


    def init_hidden(self, numlayer, numdirec=1, hidsize=200, batchsize=32):
        result = (autograd.Variable(torch.zeros(numlayer*numdirec, batchsize, hidsize).type(self.dtype)),
                  autograd.Variable(torch.zeros(numlayer*numdirec, batchsize, hidsize).type(self.dtype)))
        return result

    def init_weights(self,init_type,init_scale):
        # Initialize weight matrix
        for p in self.parameters():
            if init_type=="orthogonal" and p.dim()>=2:
                nninit.orthogonal(p)
            elif init_type=="uniform":
                p.data.uniform_(-init_scale, init_scale)
            elif init_type=="xavier_n" and p.dim()>=2:
                nninit.xavier_normal(p)
            elif init_type=="xavier_u" and p.dim()>=2:
                nninit.xavier_uniform(p)

    def forward(self, sub_embed_ind):
        # get sequence length = num of steps for this batch
        sequence_length = sub_embed_ind.size(1)

        #### C2W
        sub_batch = sub_embed_ind.view(self.batch_size*sequence_length,-1)
        sub_embeds = self.subEmbeddings(sub_batch)

        # Initialize composition hidden states with zeros
        # We do it here because sequence length changes for each batch!
        self.comp_hidden = self.init_hidden(numlayer=self.sub_num_layers, numdirec=2, hidsize=self.sub_rnn_size, \
                                            batchsize=(self.batch_size * sequence_length))
        _, self.comp_hidden = self.comp_lstm(sub_embeds, self.comp_hidden)
        h_n_f = self.comp_hidden[0][0]
        h_n_b = self.comp_hidden[0][1]
        exp_bias = self.we_bias.unsqueeze(0)
        exp_bias = exp_bias.expand(self.batch_size*sequence_length, self.word_dim)
        word_embeds = torch.mm(h_n_f,self.W_f)+torch.mm(h_n_b,self.W_b)+exp_bias
        word_embeds = word_embeds.view(self.batch_size, sequence_length, -1)
        return word_embeds
