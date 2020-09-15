from __future__ import print_function
from comet_ml import Experiment
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.small_cnn import *
from trades import trades_loss

# perform attack during training:
from PGD_on_the_fly_attack_correct import *

parser = argparse.ArgumentParser(description='PyTorch MNIST TRADES Adversarial Training')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
          
parser.add_argument('--model-dir', default='trained_models/mnist/',
                    help='directory of model for saving checkpoint')

# specify GPU id
parser.add_argument('--cuda', default=0, type=int)

# load the model, which we would like to evaluate
parser.add_argument('--size', default='SmallCNN', type=str)
parser.add_argument('--mode', default='baseline', type=str)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')

## These parameters will be used to load models as well
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--alpha', default=0.0, type=float,
                    help='margin/margin_decor')
parser.add_argument('--epsilon', default=0.3, type=float,
                    help='perturbation')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# pgd loss, we use this to compute weighted loss
parser.add_argument('--pgd_loss', default='margin', type=str)
parser.add_argument('--alpha_pgd', default=0.1, type=float,
                    help='margin/margin_decor')

# specify attack parameters
parser.add_argument('--epsilon_eval', default=0.3, type=float,
                    help='perturbation')
parser.add_argument('--num_steps_eval', default=40, type=int,
                    help='perturb number of steps')
parser.add_argument('--step_size_eval', default=0.01, type=float,
                    help='perturb step size')

# specify random data distribution for evaluation
parser.add_argument('--repeat', default=20, type=int, help='repeat times for random distribution')
parser.add_argument('--random_init',default=True, help='initialize perturbations with random numbers')
  
args = parser.parse_args()
print(args)
if args.size == 'SmallCNN':
    model_size = 'smallcnn'

# we then, load the attacker model. This model is used to generate cheating distribution
if args.mode == "baseline":
    model_name = model_size + "_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_seed_' + str(args.seed) 
    # + '_epsilon_' + str(args.epsilon)

elif args.mode == "margin":
    model_name = model_size + "_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed) 
    # + '_epsilon_' + str(args.epsilon)

# settings
if args.mode != 'IAAT' and args.mode != 'AT':
    print("We are evaluating TRADES!!!")
    model_dir = args.model_dir
    load_dir = os.path.join(model_dir, args.mode)
    load_dir = os.path.join(load_dir, model_name)

use_cuda = not args.no_cuda and torch.cuda.is_available()
seed = args.seed
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device('cuda:'+str(args.cuda))
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.ToTensor()),
                   batch_size=args.test_batch_size, shuffle=False, **kwargs)

def main():
    # init model, Net() can be also used here for training

    if args.size == 'SmallCNN':
        torch.manual_seed(seed)
        # model = SmallCNN().to(device)
        model = SmallCNN()
        model = torch.nn.DataParallel(model).cuda()

    # model to generate cheating distribution
    if args.mode == "IAAT":
        # model_name = '/cmlscratch/huiminz1/workspace/IAAT/results/train_adaptive/adv_training/checkpoint.pth'
        model_name = '/cmlscratch/huiminz1/workspace/IAAT/results/train_adaptive/adv_training/checkpoint.pth'
        
        checkpoint = torch.load(model_name)
        # print("checkpoint: ", checkpoint['model'].keys())
        # print()
        # print("model: ", model.state_dict().keys())
        model.load_state_dict(checkpoint['model'])
        print("we are evaluating: IAAT!!")

    elif args.mode == "AT":
        # model_name = '/cmlscratch/huiminz1/workspace/IAAT/results_epsilon_0.00784313725490196/train/adv_training/alpha_'+str(args.alpha)+'/checkpoint.pth'
        model_name = '/cmlscratch/huiminz1/workspace/IAAT/results_mnist_epsilon_0.3/train/adv_training/alpha_'+str(args.alpha)+'/checkpoint.pth'
        # model_name = '/cmlscratch/huiminz1/workspace/IAAT/results_epsilon_0.06274509803921569/train/adv_training/alpha_'+str(args.alpha)+'/checkpoint.pth'
        # model_name = '/cmlscratch/huiminz1/workspace/IAAT/results/train/adv_training/checkpoint.pth'
        
        checkpoint = torch.load(model_name)
        # print("checkpoint: ", checkpoint['model'].keys())
        # print()
        # print("model: ", model.state_dict().keys())
        model.load_state_dict(checkpoint['model'])
        print("we are evaluating: Adversarial Training!! ", model_name)
        
    else:
        model_name = os.path.join(load_dir, "last.checkpoint")
        checkpoint = torch.load(model_name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("we are evaluating: ", model_name)

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        acc_clean, acc_adv, acc_meaned_clean = eval_adv_test_whitebox(model, device, test_loader, args)
        print("pgd loss: {}, ACC_CLEAN: {:.2f}%, ACC_ADV: {:.2f}%".format(args.pgd_loss, acc_clean*100, acc_adv*100))
        print("acc clean traditional: {:.2f}%".format(acc_meaned_clean*100))
    else:
        acc_clean = 0
        acc_adv = 0

if __name__ == '__main__':
    main()