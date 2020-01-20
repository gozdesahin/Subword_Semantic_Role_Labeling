#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin
Main training code
"""

import time
import argparse
import prewordvectors as w2v
from loader import *
from model import *
from optim import *
from scorer import *


def roleCriterion(roleSize, wp):
    """
    Simple weighted negative log likelihood criterion
    Note that size_average should be False if run_epoch function divides loss by the batch_size
    (number of training samples)
    :param roleSize: number of labels
    :param wp: weights for each label (should be in the same order, default is 1 for each)
    :return: crit: NLL loss function
    """
    weight = torch.ones(roleSize)
    # try to weight it 0.3 or something
    weight[constants.PAD] = wp
    crit = nn.NLLLoss(weight, size_average=False)
    return crit


def run_epoch(m, data, optimizer, use_cuda=True, eval=False):
    """
    Run one epoch (either in training or evaluation mode)
    :param m: model
    :param data: data ready to be processed (acquired from dataset class)
    :param optimizer: already initialized/updated optimizer
    :param use_cuda: default True, (should be tested for CPU)
    :param eval: if True, loss will be backpropogated
    :return: avg_loss: loss averaged over batch size
    """
    # if in eval mode, there will be no dropout
    if eval:
        m.eval()
    else:
        m.train()

    costs = 0.0
    criterion = roleCriterion(m.role_size, m.wp)
    if use_cuda:
        criterion.cuda()

    # shuffle batch orders
    batchOrder = torch.randperm(len(data))

    for i in range(len(data)):
        batchIdx = batchOrder[i]
        batch = data[batchIdx]
        # clean history (hidden states are initialized inside forward pass)
        m.zero_grad()
        log_probs = m(batch[0])
        if eval:
            log_probs = Variable(log_probs.data, requires_grad=False, volatile=True).contiguous()

        # calculate loss for semantic roles for sure
        gold_labels = batch[1].view(log_probs.size(0))
        loss = criterion(log_probs, gold_labels).div(m.batch_size)
        # costs += loss.data[0]
        costs += loss.item()
        if not eval:
            # go backwards and update weights
            loss.backward()
            optimizer.step()
    avg_loss = costs/len(data)
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='train.py')

    ## Data options
    parser.add_argument('-train_file', required=False, help='Path to the training file')
    parser.add_argument('-val_file', required=False, help='Path to the validation file')
    parser.add_argument('-lang', required=False, default="tur", help='Language')

    ### Experiment Options
    parser.add_argument('-output', '-o', type=str, default='train.log', help='Output log file')
    parser.add_argument('-save_dir', default='model_srl', help='Everything will be saved here')
    parser.add_argument('-save_states', type=str, default="true", help='True if you want model files to be saved')

    ### Word Embedding Options
    parser.add_argument('-pre_word_vecs', default=None, help="If a valid path is specified, then this will load pretrained word embeddings.")
    parser.add_argument('-w2vtype', default='w2v', help="[glove|sskip|w2v|fasttext]")
    parser.add_argument('-fixed_embed', dest='fixed_embed', default=False, action='store_true', help='If True, word embeddings will not be fine tuned')
    parser.add_argument('-word_vec_size', type=int, default=50, help='Word embedding size, overwritten by supplied pre_word_vecs size')

    ### Optimization options
    parser.add_argument('-param_init_type', type=str, default="orthogonal", help='Options are [orthogonal|uniform|xavier_n|xavier_u]')
    parser.add_argument('-init_scale', type=float, default=0.05, help='If init type is uniform init weights between -x,+x')
    parser.add_argument('-bias_init', type=float, default=-3.0, help='initialization for transform gates')
    parser.add_argument('-optim', default='adadelta', help='Optimization method. [sgd|adagrad|adadelta|adam]')
    parser.add_argument('-grad_clip', type=float, default=1, help='If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm')
    parser.add_argument('-dropout', type=float, default=0.5, help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
    parser.add_argument('-decay_rate', type=float, default=0.3,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) F1 does not increase on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=450,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-patience', type=int, default=3,
                        help='the number of iterations allowed before decaying the '
                             'learning rate if there is no improvement on dev set')

    ### Runtime
    parser.add_argument('-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('-gpuid', type=int, default=0, help='Id of the GPU to run')

    ### Subword model arguments
    parser.add_argument('-sub_rnn_size', type=int, default=200, help='Size of LSTM hidden state in sub2word model if composition is bi-lstm')
    parser.add_argument('-sub_num_layers', type=int, default=1, help='Number of layers in LSTM,  if composition is bi-lstm')
    parser.add_argument('-sub_model', type=str, default='lstm', help='rnn, gru, or lstm')
    parser.add_argument('-unit', type=str, default=None, help='char, char-ngram, morpheme, word, oracle or oracle-db')
    parser.add_argument('-composition', type=str, default=None, help='none(word) or bi-lstm')
    parser.add_argument('-lowercase', dest='lowercase', action='store_true', help='lowercase data', default=False)
    parser.add_argument('-SOS', type=str, default='false', help='start of sentence symbol')
    parser.add_argument('-EOS', type=str, default='true', help='end of sentence symbol')
    parser.add_argument('-ngram', type=int, default=3, help='ngrams for units parameter')
    parser.add_argument('-char_dim', type=int, default=200, help='dimension of char embedding (for C2W model only)')
    parser.add_argument('-morph_dim', type=int, default=200, help='dimension of morpheme embedding (for M2W model only)')
    parser.add_argument('-word_dim', type=int, default=200, help='dimension of word embedding')

    ### SRL Model parameters
    parser.add_argument('-layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('-numdir', type=int, default=2, help='Number of directions')
    parser.add_argument('-hidden_size', type=int, default=128, help='Size of LSTM hidden states')
    parser.add_argument('-wp', type=float, default=1, help='Weighting for non semantic roles')

    ### Discrete Feature parameters
    parser.add_argument('-use_region_mark', type=bool, default=False,
                        help='If True, predicate context of window three will be marked')
    parser.add_argument('-use_binary_mask', type=bool, default=True,
                        help='If True, only the predicate will be marked')
    parser.add_argument('-max_seq_length', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('-cont', type=str, default='false',
                        help='continue training')
    parser.add_argument('-predicted', type=str, default='false',
                        help='use predicted morphological tags if true')

    opt = parser.parse_args()

    # check cuda
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    otype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    opt.dtype = dtype
    opt.otype = otype
    opt.use_cuda = use_cuda
    if use_cuda:
        torch.cuda.set_device(opt.gpuid)

    localtest = False
    if localtest:
        opt.train_file = '/home/sahin/Workspace/Projects/crop-rotate-augment-SRL/data/tur/development.txt'
        opt.val_file = '/home/sahin/Workspace/Projects/crop-rotate-augment-SRL/data/tur/development.txt'
        #opt.pre_word_vecs = '/home/sahin/Workspace/Projects/dataset_compilation/saved_embeddings/extrinsic_lower/tr/fasttext/final_embeds.vec'
        opt.lang = "tur"
        opt.save_dir = "./temp"
        #opt.word_vec_size = 300
        opt.param_init_type = "orthogonal"
        opt.init_scale = 0.01
        opt.optim = 'sgd'
        opt.grad_clip = 2
        opt.dropout = 0.5
        opt.learning_rate = 1
        opt.decay_rate = 0.5
        opt.epochs = 20
        opt.sub_rnn_size = 32
        opt.sub_num_layers = 1
        opt.unit = 'char-ngram'
        opt.composition = 'bi-lstm'
        opt.ngram = 3
        opt.char_dim = 32
        opt.morph_dim = 32
        opt.word_dim = 32
        opt.layers = 1
        opt.numdir = 2
        opt.hidden_size = 64
        opt.wp = 1
        opt.batch_size = 32
        opt.max_seq_length = 200


    train(opt)

def train(opt):
    start = time.time()
    save_dir = opt.save_dir
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)
    opt.eos = ''
    opt.sos = ''
    if opt.EOS == "true":
        opt.eos = '</s>'

    if opt.SOS == "true":
        opt.sos = '<s>'

    opt.use_predicted = True if opt.predicted=="true" else False

    if opt.pre_word_vecs != None:
        if opt.w2vtype in ['glove', 'sskip', 'w2v']:
            zipname = None
            # Only load the first 500K words
            maxvocsize = None
            w2i, ems = w2v.loadw2v(opt.pre_word_vecs, opt.word_vec_size, myzipfile=zipname, maxvoc=maxvocsize)
            if opt.word_vec_size != len(ems[0]):
                opt.word_vec_size = len(ems[0])
        elif opt.w2vtype=='fasttext':
            w2i, ems = w2v.loadft(opt.pre_word_vecs)
            opt.word_vec_size = 300
    else:
        ems = None

    # word indexer depends on the vocabulary
    if (opt.pre_word_vecs != None) and (opt.w2vtype!='fasttext'):
        word_to_ix = w2i
    else:
        # fasttext will handle OOV words, so do not fill a fixed vocabulary
        word_to_ix = None

    # write log to train.log
    fout = open(os.path.join(opt.save_dir, opt.output), "a")
    fout.write(str(opt) + "\n")

    # load training and validation data
    ldr = Loader(opt, opt.train_file, opt.val_file, save_dir=opt.save_dir, train=True, w2i=word_to_ix)

    training_data = ldr.getData(ldr.train_data, train=True)
    validation_data = ldr.getData(ldr.dev_data, train=False)

    opt.vocab_size = len(ldr.word_to_ix)
    opt.role_size = len(ldr.role_to_ix)
    if opt.unit != "word":
        opt.subword_vocab_size = ldr.subloader.subword_vocab_size

    # Statistics of words
    fout.write("Word vocab size: " + str(opt.vocab_size) + "\n")
    fout.write("Role size: " + str(opt.role_size) + "\n")

    # Statistics of sub units

    fout.write("Unit: " + opt.unit + " Composition: " + opt.composition + "\n")

    if opt.unit != "word":
        fout.write("Subword vocab size: " + str(ldr.subloader.subword_vocab_size) + "\n")
        if opt.composition == "bi-lstm":
            if opt.unit == "char":
                fout.write("Maximum word length: " + str(ldr.subloader.max_word_len) + "\n")
            elif opt.unit == "char-ngram":
                fout.write("Maximum ngram per word: " + str(ldr.subloader.max_ngram_per_word) + "\n")
            elif opt.unit == "morpheme" or opt.unit == "oracle":
                fout.write("Maximum morpheme per word: " + str(ldr.subloader.max_morph_per_word) + "\n")
            else:
                sys.exit("Wrong unit.")
        else:
            sys.exit("Wrong unit/composition.")
    else:
        if opt.composition != "none":
            sys.exit("Wrong composition.")

    with open(os.path.join(opt.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(opt, f)

    print("Begin training...")

    # Create a model with user options
    mtrain = SRL(opt,ems)
    if opt.use_cuda:
        mtrain = mtrain.cuda()

    nParams = sum([p.nelement() for p in mtrain.parameters()])
    print('* number of parameters: %d' % nParams)

    optim = Optim(
        opt.optim, opt.learning_rate, opt.grad_clip,
        lr_decay=opt.decay_rate,
        patience=opt.patience
    )
    # if word embeddings will be fixed, do not update them
    if opt.fixed_embed:
        optim.set_parameters(filter(lambda p: p.requires_grad, mtrain.parameters()))
    else:
        optim.set_parameters(mtrain.parameters())

    if opt.cont == 'true':  # continue training from a saved model
        # get model parameters
        model_path, e = get_last_model_path(opt.save_dir)
        mtrain = torch.load(model_path)
    else:
        # process each epoch
        e = 1

    bestF1 = 0.0

    while e <= opt.epochs:
        print("Epoch: %d " % (e))
        print("Learning rate: %.3f" % optim.lr)

        #  (1) train for one epoch on the training set
        train_loss = run_epoch(mtrain, training_data, optim, use_cuda=opt.use_cuda, eval=False)
        print("Train Loss: %.3f" % train_loss)

        #  (2) evaluate on the validation set
        dev_cur_loss = run_epoch(mtrain, validation_data, optim, use_cuda=opt.use_cuda, eval=True)
        print("Valid Loss: %.3f" % dev_cur_loss)

        _, _, num_corr_sr, num_found_sr, num_gold_sr = testRoleLabels(mtrain, validation_data, ldr.role_to_ix,
                                                                               mode="train", type="simple")

        f1 = writeScores(num_corr_sr, num_found_sr, num_gold_sr, fout)
        print("F1: ",f1)

        #  (3) update the learning rate
        optim.updateLearningRate(f1, e)

        # (4) save results and report
        diff = f1 - bestF1
        if diff >= 0.01:
            if opt.save_states=="true":
                torch.save(mtrain,'%s/%s-%d.pt' % (save_dir, "model", e))
            bestF1 = f1

        # write results to file
        fout.write("Epoch: %d\n" % e)
        fout.write("Learning rate: %.3f\n" % optim.lr)
        fout.write("Train Perplexity: %.3f\n" % train_loss)
        fout.write("Valid Perplexity: %.3f\n" % dev_cur_loss)
        fout.write("F1: %.3f\n" % f1)
        fout.flush()

        if optim.lr < 0.0001:
            print('Learning rate too small, stop training.')
            break

        e += 1

    print("Training time: %.0f" % (time.time() - start))
    fout.write("Training time: %.0f\n" % (time.time() - start))
    print("Cleaning")
    remove_except_last_model(opt.save_dir)

if __name__ == "__main__":
    main()