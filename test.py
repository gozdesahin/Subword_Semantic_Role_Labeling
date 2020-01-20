#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Test the file given the trained model
"""

import subprocess

from IO.conllWriter import *
from IO.util import *
from loader import *
from scorer import *
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_file', type=str, default='data/CoNLL2009-ST-Turkish/CoNLL2009-ST-evaluation-Turkish.txt',
                        help="test file")
    parser.add_argument('-save_dir', type=str, default='model_srl',
                        help='directory of the checkpointed models')
    parser.add_argument('-lang', type=str, default='tur',
                        help='language of test file')
    parser.add_argument('-dt', type=str, default='test',
                        help='data type (test|dev|ood)')
    parser.add_argument('-gpuid', type=int, default=0, help='Id of the GPU to run')
    args = parser.parse_args()

    localtest = False
    if localtest:
        args.test_file = '/home/sahin/Workspace/Projects/crop-rotate-augment-SRL/data/tur/evaluation.txt'
        args.save_dir = './temp'
        args.lang = 'tur'
        args.dt = 'test'

    test(args)

def test(test_args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(test_args.gpuid)

    with open(os.path.join(test_args.save_dir, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)

    args.save_dir = test_args.save_dir
    args.batch_size = 1
    dt = test_args.dt
    ldr = Loader(args, test_file=test_args.test_file, save_dir = test_args.save_dir, train=False, test=True)
    goldFilePath = test_args.test_file
    test_data = ldr.getData(ldr.test_data, train=False)

    print("Begin testing...")
    # get the last saved model
    model_path, _ = get_last_model_path(args.save_dir)
    mtest = torch.load(model_path)

    if args.use_cuda:
        mtest = mtest.cuda()

    # change all batch sizes to 1
    mtest.batch_size = 1
    if mtest.subwordModel != None:
        mtest.subwordModel.batch_size=1

    predictedSenseSents = None
    plst, glst, num_corr_sr, num_found_sr, num_gold_sr = testRoleLabels(mtest, test_data, ldr.role_to_ix,
                                                                           mode="eval", type="simple")

    systemFilePath = os.path.join(test_args.save_dir, ("system_"+dt+".conll"))


    conllOut = codecs.open(systemFilePath, "w", encoding='utf-8')
    if (test_args.lang=="fin"):
        writeCoNLLUD(conllOut, ldr, plst, predictedSenseSents)
    else:
        writeCoNLL(conllOut, ldr, plst, predictedSenseSents)

    # necesary for copula handling in conll09 files
    if (test_args.lang in ["tur", "fin"]):
        goldFilePath = os.path.join(test_args.save_dir, ("goldTest_"+dt+".conll"))
        goldConllOut = codecs.open(os.path.join(test_args.save_dir, ("goldTest_"+dt+".conll")), "w", encoding='utf-8')
        if (test_args.lang=="fin"):
            writeCoNLLUD(goldConllOut, ldr, glst)
        else:
            writeCoNLL(goldConllOut, ldr, glst)

    # run eval09 script
    scoreOut = codecs.open(os.path.join(test_args.save_dir, ("eval09_analysis_"+dt+".conll")), "w", encoding='utf-8')
    subprocess.call(["perl", "eval09.pl","-g",goldFilePath ,"-s" ,systemFilePath], stdout=scoreOut)

    # run self evaluator and write to test.log file
    log_out = open(os.path.join(test_args.save_dir, ("test_scores_"+dt+".log")), "w")
    writeScores(num_corr_sr, num_found_sr, num_gold_sr, log_out)

if __name__ == '__main__':
    main()