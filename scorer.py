#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Evaluate predicted labels by comparing to gold labels

"""

import torch
from itertools import chain

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def buildi2s(s2i):
    i2s = {}
    for key in s2i:
        i2s[s2i[key]]=key
    return i2s


def evalConll(pred_labels, gold_labels, dummyrole, stoprole, role_to_ix, mode, type):
    """
    Run one epoch (either in training or evaluation mode)
    :param pred_labels: predicted labels
    :param gold_labels: ground truth
    :param dummyrole: "_", don't score it
    :param stoprole: <STOPTAG> if using CRF sequence model don't score it
    :param mode: if "eval", save the predicted and gold labels into a list
    :param type: if "seq", remove the start and stop tag
    :return: plst: predicted label list
    :return: glst: gold label list
    :return: num_corr,num_found,num_gold : correctly found, total found argument count, total gold
    """
    num_corr = 0
    num_gold = 0
    num_found = 0
    # local lists
    plst = []
    glst = []
    index2role = buildi2s(role_to_ix)
    index2role[0] = u"_"
    for i in range(pred_labels.size(0)):
        found = pred_labels[i,:].contiguous().view(1,-1)
        if found.type() != 'torch.cuda.FloatTensor':
            found = found.float()
        if mode=="eval":
            fstr=""
            for ind in found[0]:
                fstr+=index2role[int(ind)]
                fstr+=" "
            plst.append(fstr[:-1])
        if (type=="seq"):
            # remove the start and the stop tag
            gold = gold_labels[i,1:-1].contiguous().view(1,-1)
        else:
            gold = gold_labels[i,:].contiguous().view(1, -1)
        if mode=="eval":
            gstr=""
            for ind in gold[0]:
                gstr+=index2role[int(ind)]
                gstr+=" "
            glst.append(gstr[:-1])
        label_mask = gold.ne(dummyrole)&gold.ne(stoprole)

        is_corr = (found==gold)*label_mask
        num_corr += is_corr.ne(0).sum()
        num_gold += label_mask.sum()
        num_found += (found.ne(dummyrole)&found.ne(stoprole)).sum()
    return plst,glst,num_corr,num_found,num_gold

def testRoleLabels(model, data, role_to_ix, mode="eval", type="simple"):
    """
    Basic Inference Code
    Given the trained model and gold labels, calculate model's best semantic role predictions and compare to gold
    :param model: SRL model
    :param data: test data
    :param role_to_ix: semantic role dictionary
    :param mode:
    :param type: can be ignored
    :return:
    """
    total_corr = 0.
    total_found = 0.
    total_gold = 0.

    predictionslst = []
    goldlst = []

    # get index of non-roles
    dummyrole = role_to_ix['_']

    if mode == "eval":
        model.eval()
    for i in range(len(data)):
        batch = data[i]

        modscore = model(batch[0])

        _, mod_tag_seq = torch.max(modscore, 1)
        mod_tag_seq = mod_tag_seq.data
        mod_tag_seq = mod_tag_seq.view(model.batch_size, -1)

        gold_lab = batch[1]
        plst, glst, num_corr, num_found, num_gold = evalConll(mod_tag_seq, gold_lab.data.type(dtype), dummyrole,
                                                              dummyrole, role_to_ix, mode, type)
        predictionslst.append(plst)
        goldlst.append(glst)
        total_corr += num_corr
        total_gold += num_gold
        total_found += num_found

    predictionslst = list(chain.from_iterable(predictionslst))
    goldlst = list(chain.from_iterable(goldlst))

    return predictionslst, goldlst, total_corr, total_found, total_gold

def testRoleLabelsEnsemble(models, datas, role_to_ix, mode="eval", type="simple"):
    """
    Given multiple models, calculate predictions via average voting and compare to gold
    """
    total_corr = 0.
    total_found = 0.
    total_gold = 0.

    predictionslst = []
    goldlst = []

    # get index of non-roles
    dummyrole = role_to_ix['_']

    if mode=="eval":
        for model in models:
            model.eval()

    for i in range(len(datas[0])):
        log_probs = torch.zeros(datas[0][i][1].size(1), len(role_to_ix)).cuda()
        # averaging - voting
        for k, model in enumerate(models):
            batch = datas[k][i]
            lp = model(batch[0]).data
            log_probs += lp

        avg_log_probs = torch.autograd.Variable(torch.div(log_probs, len(models)), volatile=True)
        _, mod_tag_seq = torch.max(avg_log_probs, 1)
        mod_tag_seq = mod_tag_seq.data
        mod_tag_seq = mod_tag_seq.view(1, -1)

        # gold labels
        plst, glst, num_corr, num_found, num_gold = evalConll(mod_tag_seq, datas[0][i][1].data.type(dtype), dummyrole,
                                                              dummyrole, role_to_ix, mode, type)
        predictionslst.append(plst)
        goldlst.append(glst)
        total_corr += num_corr
        total_gold += num_gold
        total_found += num_found

    predictionslst = list(chain.from_iterable(predictionslst))
    goldlst = list(chain.from_iterable(goldlst))

    return predictionslst, goldlst, total_corr, total_found, total_gold

def testRoleLabelsEnsembleLearner(models, ensmodel, datas, role_to_ix, mode="eval", type="simple"):
    """
    Stack Generalizer (ensmodel) learns to weigh the predictions of other SRL models
    Calculate SG's prediction given other models' predictions
    :param models: Pretrained SRL models
    :param ensmodel: Stack Generalizer model
    """
    total_corr = 0.
    total_found = 0.
    total_gold = 0.

    predictionslst = []
    goldlst = []

    # get index of non-roles
    dummyrole = role_to_ix['_']

    if mode=="eval":
        for model in models:
            model.eval()
        ensmodel.eval()

    for i in range(len(datas[0])):
        log_probs = []
        for z, model in enumerate(models):
            batch = datas[z][i]
            lp = model(batch[0])
            log_probs += [lp]
        input = torch.stack(log_probs)

        final_log_probs = ensmodel(input)
        _, mod_tag_seq = torch.max(final_log_probs, 1)
        mod_tag_seq = mod_tag_seq.data
        mod_tag_seq = mod_tag_seq.view(1, -1)

        # gold labels
        plst, glst, num_corr, num_found, num_gold = evalConll(mod_tag_seq, datas[0][i][1].data.type(dtype), dummyrole,
                                                              dummyrole, role_to_ix, mode, type)
        predictionslst.append(plst)
        goldlst.append(glst)
        total_corr += num_corr
        total_gold += num_gold
        total_found += num_found

    predictionslst = list(chain.from_iterable(predictionslst))
    goldlst = list(chain.from_iterable(goldlst))

    return predictionslst, goldlst, total_corr, total_found, total_gold