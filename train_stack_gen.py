#!/usr/bin/env python
# -*- coding: utf-8 -*-
from IO.util import *
from SGModel import *
from loader import *
from optim import *
from scorer import *
import argparse


def main():
    parser = argparse.ArgumentParser(description='train_stack_gen.py')

    ## Data options
    # Ensembler will be learned on validation file
    parser.add_argument('-val_file', required=False, help='Path to the validation file')
    # base learner models
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
    parser.add_argument('-lang', required=False, default="tr", help='Language (en|tr)')

    ### Experiment Options
    parser.add_argument('-output', '-o', type=str, default='enstrain.log', help='Output log file')
    parser.add_argument('-save_dir', default='ensem_model', help='Everything will be saved here')
    parser.add_argument('-cross_validation', dest='cross_validation', default=False, action='store_true', help='If True, training_data will be divided into k')
    parser.add_argument('-k', type=int, default=5, help='k-fold cross validation')
    parser.add_argument('-save_states', type=str, default="true", help='True if you want model files to be saved')

    ### Optimization options
    parser.add_argument('-param_init_type', type=str, default="orthogonal", help='Options are [orthogonal|uniform|xavier_n|xavier_u]')
    parser.add_argument('-init_scale', type=float, default=0.05, help='If init type is uniform init weights between -x,+x')
    parser.add_argument('-optim', default='adadelta', help='Optimization method. [sgd|adagrad|adadelta|adam]')
    parser.add_argument('-grad_clip', type=float, default=1, help='If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm')
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
    parser.add_argument('-decay_rate', type=float, default=0.3)
    parser.add_argument('-start_decay_at', type=int, default=450)
    parser.add_argument('-patience', type=int, default=3,
                        help='the number of iterations allowed before decaying the '
                             'learning rate if there is no improvement on dev set')

    ### Runtime
    parser.add_argument('-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('-gpuid', type=int, default=0, help='Id of the GPU to run')

    ### Ensemble Model parameters
    parser.add_argument('-indim', type=int, default=2, help='Number of models to ensemble')
    parser.add_argument('-hiddim', type=int, default=2, help='Hidden size')
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
    train(opt)

def get_models(experiments, devfile):
    basel_lrnr_lst = []
    data_lst = []
    role_to_ix = {}

    for model_dir in experiments:
        if model_dir==None:
            break
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            args = pickle.load(f)

        args.save_dir = model_dir

        ldr = Loader(args, test_file=devfile, save_dir = model_dir, train=False, test=True)

        if len(role_to_ix)==0:
            role_to_ix = ldr.role_to_ix
            test_data = ldr.getData(ldr.test_data, train=True)
        else:
            test_data = ldr.getData(ldr.test_data, train=False)

        model_path, _ = get_last_model_path(model_dir)
        mtest = torch.load(model_path)
        if args.use_cuda:
            mtest = mtest.cuda()

        mtest.eval()
        basel_lrnr_lst.append(mtest)
        data_lst.append(test_data)
    return basel_lrnr_lst,data_lst,role_to_ix

def train(opt):
    save_dir = opt.save_dir
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    fout = open(os.path.join(opt.save_dir, opt.output), "a")

    experiments = [opt.save_dir1,opt.save_dir2,opt.save_dir3,opt.save_dir4, \
                   opt.save_dir5,opt.save_dir6,opt.save_dir7]
    devFile = opt.val_file
    optim = Optim(
        opt.optim, opt.learning_rate, opt.grad_clip,
        lr_decay=opt.decay_rate,
        patience=opt.patience
    )
    models, datas, role_to_ix = get_models(experiments, devFile)
    # mtrain = EnsModel(opt.indim, len(role_to_ix), opt)
    mtrain = SGModelSimple(opt.indim, len(role_to_ix), opt)

    if opt.use_cuda:
        mtrain = mtrain.cuda()
    optim.set_parameters(mtrain.parameters())
    criterion = nn.NLLLoss()
    if use_cuda:
        criterion.cuda()

    print "Begin training..."
    e = 0

    numFolds = opt.k
    size = len(datas[0])/numFolds
    chunk_inds = [range(i*size,(i+1)*size) for i in xrange(numFolds)]
    #prev_val_costs = [10000000]*numFolds
    best_val_cost = 10000000
    while e <= opt.epochs:
        print("Epoch: %d" % (e))
        training_cost = 0.0
        val_cost = 0.0
        k = 2 #e % numFolds
        # Calculate training cost
        for i in range(len(datas[0])):
            # find fold number
            if i in chunk_inds[k]:
                evalMode = True
            else:
                evalMode = False
            if evalMode:
                mtrain.eval()
            log_probs = []
            mtrain.zero_grad()
            # get all predictions from each method
            for z, model in enumerate(models):
                batch = datas[z][i]
                lp = model(batch[0])
                log_probs += [lp]
            input = torch.stack(log_probs)
            if not evalMode:
                input = Variable(input.data, requires_grad=True, volatile=False).contiguous()
            new_labels = mtrain(input)
            gold_labels = datas[0][i][1]
            loss = criterion(new_labels, gold_labels.view(-1))
            if not evalMode:
                training_cost += loss.data[0]
                # go backwards and update weights
                loss.backward()
                optim.step()
            else:
                val_cost += loss.data[0]
        # Calculate development cost
        print ("Training Cost: %f" % (training_cost))
        print ("Validation Cost: %f" % (val_cost))
        fout.write("Epoch: %d\n" % e)
        fout.write("Learning rate: %.3f\n" % optim.lr)
        fout.write("Train Perplexity: %.3f\n" % training_cost)
        fout.write("Valid Perplexity: %.3f\n" % val_cost)
        if (val_cost < best_val_cost):
            # save the ensemble weights
            torch.save(mtrain, '%s/%s-%d.pt' % (save_dir, "ensweight", e))
            best_val_cost = val_cost
        e+=1

if __name__ == "__main__":
    main()