#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin
Ensemble models by averaging
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
    parser.add_argument('-save_dir1', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-save_dir2', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-save_dir3', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-save_dir4', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-save_dir5', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-save_dir6', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-save_dir7', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-ens_save_dir', required=False, default=None,
                        help="directory of the checkpointed models")
    parser.add_argument('-lang', type=str, default='tur',
                        help='directory of the checkpointed models')
    parser.add_argument('-gpuid', type=int, default=0, help='Id of the GPU to run')
    args = parser.parse_args()
    test(args)

def test(test_args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(test_args.gpuid)

    # global settings
    goldFile = test_args.test_file

    experiments = [test_args.save_dir1,test_args.save_dir2,test_args.save_dir3,test_args.save_dir4, \
                   test_args.save_dir5,test_args.save_dir6,test_args.save_dir7]

    predictedSenseSents = None

    try:
        os.stat(test_args.ens_save_dir)
    except:
        os.mkdir(test_args.ens_save_dir)

    models_lst = []
    test_data_lst = []
    role_to_ix = {}
    ldr_gen = None

    for model_dir in experiments:
        if model_dir==None:
            break
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            args = pickle.load(f)

        args.save_dir = model_dir
        args.batch_size = 1

        ldr = Loader(args, test_file=goldFile, save_dir = model_dir, train=False, test=True)

        if len(role_to_ix)==0:
            role_to_ix = ldr.role_to_ix
            ldr_gen = ldr

        test_data = ldr.getData(ldr.test_data, train=False)
        model_path, _ = get_last_model_path(model_dir)
        mtest = torch.load(model_path)
        if args.use_cuda:
            mtest = mtest.cuda()

        # change all batch sizes to 1
        mtest.batch_size = 1
        if mtest.subwordModel != None:
            mtest.subwordModel.batch_size = 1

        models_lst.append(mtest)
        test_data_lst.append(test_data)

    print("Begin testing...")
    plst, glst, num_corr_sr, num_found_sr, num_gold_sr = testRoleLabelsEnsemble(models_lst, test_data_lst, role_to_ix,
                                                                           mode="eval", type="simple")
    # Write results
    systemFilePath = os.path.join(test_args.ens_save_dir, "system.conll")


    conllOut = codecs.open(systemFilePath, "w", encoding='utf-8')
    if (test_args.lang=="fin"):
        writeCoNLLUD(conllOut, ldr_gen, plst, predictedSenseSents)
    else:
        writeCoNLL(conllOut, ldr_gen, plst, predictedSenseSents)

    # necessary for copula handling in conll09 files
    if (test_args.lang in ["tur", "fin"]):
        goldFile = os.path.join(test_args.ens_save_dir, "goldTest.conll")
        goldConllOut = codecs.open(os.path.join(test_args.ens_save_dir, "goldTest.conll"), "w", encoding='utf-8')
        if (test_args.lang=="fin"):
            writeCoNLLUD(goldConllOut, ldr_gen, glst)
        else:
            writeCoNLL(goldConllOut, ldr_gen, glst)

    # run eval09 script
    scoreOut = codecs.open(os.path.join(test_args.ens_save_dir, "eval09_analysis.out"), "w", encoding='utf-8')
    subprocess.call(["perl", "eval09.pl","-g", goldFile,"-s" ,systemFilePath], stdout=scoreOut)

    # run self evaluator and write to test.log file
    log_out = open(os.path.join(test_args.ens_save_dir, "test_scores.log"), "w")
    writeScores(num_corr_sr, num_found_sr, num_gold_sr, log_out)


if __name__ == "__main__":
    main()
