#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin
Load pretrained embeddings
"""

import zipfile
import constants
import numpy as np
from gensim.models import FastText as fText

def loadw2v(embfile, embsize, myzipfile=None, maxvoc=None):
    """
    Load pretrained embeddings from text or a zip file
    Index 0 is for padding
    Index 1 is for unknowns
    :param embfile: word embedding file
    :param embsize: word vector size
    :param myzipfile: embeddings in zip file
    :param maxvoc: maximum vocabulary size
    :return: dictionary, model (numpy matrix of shape (num_embeddings, embedding_dim))
    """
    word_to_ix = {}
    word_to_ix[constants.PAD_ITEM] = 0
    word_to_ix[constants.UNK_ITEM] = 1
    # fill padding word with zeros
    model = [[0.]*embsize]
    # fill unk word with random numbers
    model.append(np.random.normal(0,0.15,size=embsize).tolist())
    if myzipfile != None:
        zip = zipfile.ZipFile(myzipfile, 'r')
        f = zip.read(embfile).split("\n")
    else:
        f = open(embfile, 'r')
    ix = 2
    for line in f:
        if maxvoc!=None:
            if ix >= maxvoc:
                break
        splitLine = line.split()
        if(len(splitLine)>embsize+1):
            phrase_lst = splitLine[:-embsize]
            word = ' '.join(phrase_lst)
            embedding = [float(val) for val in splitLine[-embsize:]]
            word_to_ix[word] = ix
            model.append(embedding)
            ix += 1
        elif(len(splitLine)>2):
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            word_to_ix[word]=ix
            model.append(embedding)
            ix += 1
        else:
            print line
    print("%d words loaded!" % len(model))
    return word_to_ix, model


def loadft(embfile):
    # load fasttext model with fixed vocab
    model = fText.load_fasttext_format(embfile)
    # this will be a filled from the training data
    word_to_ix = None
    return word_to_ix, model
