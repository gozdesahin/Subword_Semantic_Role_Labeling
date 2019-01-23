#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin

Load (morphed) conll09 dataset, create vocabularies
Calls TextLoader to get encodings of word-parts
Pack all information into a Dataset
"""

from itertools import chain

from IO.conll09 import *
from IO.conllud_fin import *
from SUB.wordpart import *
from dataset import *

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def wrap(mylist):
    return torch.FloatTensor(mylist)


def prepare_sequence(seq, to_ix):
    """
    return UNK index (1) if word not found
    :param seq: sequence
    :param to_ix: dictionary
    :return: torch tensor of indices
    """
    idxs = map(lambda w: to_ix[w] if (w in to_ix) else constants.UNK, seq)
    tensor = torch.LongTensor(idxs)
    return tensor


def prepare_label_sequence(seq, to_ix, modeltype="simple"):
    """
    Put start and stop tag for delimitation
    If it is a sequence model, we put a start and stop tag, otherwise there will be nothing
    :param seq: sequence
    :param to_ix: dictionary
    :param modeltype: simple or sequence
    :return:
    """
    idxs = []
    if modeltype=="seq":
        idxs.append(to_ix[constants.START_TAG])
    for l in seq:
        if (modeltype == "seq")|(modeltype == "simple"):
            idxs.append(to_ix[l]) if(l in to_ix) else idxs.append(constants.UNK)
        else:
            idxs.append(0) if (l=="_") else idxs.append(1)
    if modeltype=="seq":
        idxs.append(to_ix[constants.STOP_TAG])
    tensor = torch.LongTensor(idxs)
    return tensor


class Loader():
    def __init__(self, opt, train_file=None, dev_file=None, test_file=None, save_dir=None, train=True, test=False, w2i=None, r2i=None):
        """
        :param opt: argument labeling options
        :param train_file: training file
        :param dev_file: When dev_file is None, it means we work on cross-validation mode
        :param test_file: test file (in test mode)
        :param save_dir: directory to save vocabularies
        :param train: True if in training mode
        :param test: True if in testing mode
        :param w2i: None or word vocabulary from pretrained word vectors
        :param r2i: None if train, else from a previous loader
        """
        # word lookup table
        self.word_to_ix = w2i if (w2i != None) else {}
        # semantic role lookup table
        self.role_to_ix = r2i if (r2i != None) else {}
        self.options = opt
        self.c9sents = None
        self.train = train
        self.test = test
        self.ft_embeds = (opt.w2vtype == 'fasttext')

        self.role_vocab_file = os.path.join(save_dir, "role_vocab.pkl")
        self.word_vocab_file = os.path.join(save_dir, "words_vocab.pkl")

        # load conll09 file, init lookups, prepare sorted data
        if(train==True):
            opt.trainLst = self.prepare_input_for_subloader(train_file, opt.unit)
            self.subloader = TextLoader(opt, train=train)
            self.build_vocab(train_file)
            self.train_data = self.process(train_file)
            if dev_file != None:
                self.dev_data = self.process(dev_file)

        elif (test==True):
            self.subloader = TextLoader(opt, train=False)
            self.load_preprocessed()
            if test_file != None:
                self.test_data = self.process(test_file)
            if train_file != None:
                self.train_data = self.process(train_file)
            if dev_file != None:
                self.dev_data = self.process(dev_file)

    def load_preprocessed(self):
        with open(self.role_vocab_file, 'rb') as f:
            self.role_to_ix = pickle.load(f)
        with open(self.word_vocab_file, 'rb') as f:
            self.word_to_ix = pickle.load(f)

    def build_vocab(self, filepath):
        """
        Builds role and word vocabularies
        :param filepath: path to training file
        :return:
        """
        # fall back to subloader word_to_id if no pretrained embeddings are given
        if len(self.word_to_ix)==0:
            self.word_to_ix = self.subloader.word_to_id
        # save vocabulary under base folder
        with open(self.word_vocab_file, 'wb') as f:
            pickle.dump((self.word_to_ix), f)

        self.role_to_ix[constants.PAD_ROLE] = len(self.role_to_ix)
        self.role_to_ix[constants.UNK_ROLE] = len(self.role_to_ix)

        if self.c9sents == None:
            if self.options.lang=="fin":
                c9reader = conllud_fin(filepath)
            else:
                c9reader = conll09(filepath, self.options.use_predicted)
            self.c9sents = c9reader.sents
        # for each sentence
        for c9sent in self.c9sents:
            for i in xrange(c9sent.predcnt):
                # role labels
                sroles = c9sent.labels[i]
                for role in sroles:
                    if role not in self.role_to_ix:
                        self.role_to_ix[role] = len(self.role_to_ix)

        with open(self.role_vocab_file, 'wb') as f:
            pickle.dump((self.role_to_ix), f)

    def prepare_input_for_subloader(self, filepath, unit):
        """
        if unit is oracle , make a list from token_oracles, otherwise use token_words
        :param filepath: path to training file
        :param unit: subword unit
        :return:
        """
        sentLst = []
        if self.c9sents == None:
            if self.options.lang=="fin":
                c9reader = conllud_fin(filepath)
            else:
                c9reader = conll09(filepath, self.options.use_predicted)
            self.c9sents = c9reader.sents
        for c9sent in self.c9sents:
            if unit == "oracle":
                tokenLst = c9sent.tokenOracles
            else:
                tokenLst = c9sent.tokenWords
            if(len(tokenLst) <= self.options.max_seq_length):
                tokenLst = map(lambda w: w.lower(), tokenLst)
                sentLst.append(tokenLst)
        return sentLst

    def process(self, filepath):
        """
        Make a sorted list of training data a.t. their sentence length
        Packed info: word, morphological representation, predicate flag, predicate word, semantic roles
        :param filepath: path to training file
        :return: packed training data
        """
        if self.options.lang == "fin":
            c9reader = conllud_fin(filepath)
        else:
            c9reader = conll09(filepath, self.options.use_predicted)
        c9sents = c9reader.sents
        # for sorting purposes
        sentBucket = {}
        # if in evaluation mode do not sort data, there will be no batching
        if self.test:
            unsorted_data = []
            self.c9sents = c9sents
        # for each sentence
        for c9sent in c9sents:
            # tokens
            sentTokenLst = c9sent.tokenWords
            # oracles
            sentOracleLst = c9sent.tokenOracles

            if(len(sentTokenLst) <= self.options.max_seq_length):
                sentTokenLst = map(lambda w: w.lower(), sentTokenLst)
                sentOracleLst = map(lambda w: w.lower(), sentOracleLst)
                numWords = len(sentTokenLst)
                # for each predicate
                for i in xrange(c9sent.predcnt):
                    predInd = c9sent.predind[i]
                    genData = []
                    # 1) tokens
                    genData.append(sentTokenLst)
                    # 2) oracles
                    genData.append(sentOracleLst)
                    # 3) binary flag for predicate
                    binmask = [0] * numWords
                    binmask[predInd] = 1
                    genData.append(binmask)
                    # 4) predicate word
                    predWord = sentTokenLst[predInd].lower().split()
                    genData.append(predWord)
                    # 5) gold semantic roles
                    sroles = c9sent.labels[i]
                    genData.append(sroles)
                    ### sorting/batching
                    if numWords in sentBucket:
                        sentBucket[numWords].append(genData)
                    else:
                        sentBucket[numWords]=[genData]
                    if self.test:
                        unsorted_data.append(genData)
        sorted_data = list(chain.from_iterable(sentBucket.values()))
        return sorted_data if self.train else unsorted_data

    def getData(self, raw_data, train=True, modeltype="simple"):
        """

        :param raw_data: packed training data by process function
        :param train: True if in training mode
        :param modeltype: can be ignored - will be simple
        :return: dataset object - ready to batch
        """
        dataTok = []
        dataDebug = []
        dataSub = []
        dataBM = []
        dataPW = []
        dataRole = []

        for sentence, morph_anal, bmSeq, predWord, roles in raw_data:
            sentence = map(lambda w:w.lower(),sentence)
            # predicate to lower
            predWord[0] = predWord[0].lower()
            # 1) token
            # if fasttext, pass the words itself
            if self.ft_embeds:
                dataTok.append(sentence)
            else:
                word_embed_ind = prepare_sequence(sentence, self.word_to_ix)
                dataTok.append(word_embed_ind)
            word_embed_ind = prepare_sequence(sentence, self.word_to_ix)
            dataDebug.append(word_embed_ind)
            # 2) subword of your choice (comes already padded)
            if self.subloader.unit == "oracle":
                sub_encoded = self.subloader.encode_data(morph_anal)
            else:
                sub_encoded = self.subloader.encode_data(sentence)
            dataSub.append(torch.LongTensor(sub_encoded).type(self.options.otype))
            # 3) binary mask
            word_bin_feat = wrap(map(lambda reg: float(reg), bmSeq))
            dataBM.append(word_bin_feat)
            # 4) predicate word
            word_pred_ind = prepare_sequence(predWord, self.word_to_ix)
            dataPW.append(word_pred_ind)
            # 5) semantic roles
            training_labels = prepare_label_sequence(roles, self.role_to_ix, modeltype)
            dataRole.append(training_labels)

        # If in evaluation mode, do not keep the history (volatile)
        # When the model type is simple, we pad the label sequence with zero
        if train:
            dset = Dataset([dataTok, dataSub, dataBM, dataPW], [dataRole],
                           self.options.batch_size, use_cuda, self.ft_embeds)
        else:
            dset = Dataset([dataTok, dataSub, dataBM, dataPW], [dataRole],
                           self.options.batch_size, use_cuda, self.ft_embeds, volatile=True)
        del dataTok, dataBM, dataPW, dataRole
        return dset