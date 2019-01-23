#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Utility functions

"""

import torch
import sys
import os


# global definitions
#use_cuda = torch.cuda.is_available()
#dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def get_last_model_path(save_dir):
    """
    Get the last saved model
    :param save_dir: Directory where models are saved
    :return: the full path of the last model, next epoch (if we will continue training)
    """
    nexte = 2
    model_name = "model-1.pt"
    for file in os.listdir(save_dir):
        if file.endswith(".pt") and file.__contains__("-"):
            ce = int(file.split('-')[1][:-3])+1
            if ce>nexte:
                nexte = ce
                model_name = file
    full_path = os.path.join(save_dir, model_name)
    return full_path, nexte

def remove_except_last_model(save_dir):
    """
    Remove all model files except the last one
    :param save_dir: Directory where models are saved
    :return:
    """
    fullp, nexte = get_last_model_path(save_dir)
    laste = nexte
    for file in os.listdir(save_dir):
        if file.endswith(".pt") and file.__contains__("-"):
            ce = int(file.split('-')[1][:-3])+1
            if ce<laste:
                model_name = file
                full_path = os.path.join(save_dir, model_name)
                os.remove(full_path)
    return

def writeScores(total_corr, total_found, total_gold, fout):
    """
    Write precision, recall and F1 to screen and to file
    :param total_corr: Correctly found labels
    :param total_found: Total number of detected labels
    :param total_gold: Total number of gold labels
    :param fout: File to write scores
    :return:
    """
    # to pretend division by zero
    total_found = float(total_found)+sys.float_info.epsilon
    total_corr = float(total_corr)
    total_gold = float(total_gold)
    pr = total_corr/total_found
    re = total_corr/total_gold
    f1 = (2.0*pr*re)/(pr+re+sys.float_info.epsilon)

    fout.write("Argument Labeling: Precision: %.3f Recall: %.3f F1: %3f \n" % (pr, re, f1))
    print("Argument Labeling: Precision: %.3f Recall: %.3f F1: %3f " % (pr, re, f1))
    return f1