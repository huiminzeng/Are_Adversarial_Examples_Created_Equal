from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss

# perform attack during training:
from PGD_on_the_fly_attack_correct import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
          
parser.add_argument('--model-dir', default='trained_models/cifar10/',
                    help='directory of model for saving checkpoint')

# specify GPU id
parser.add_argument('--cuda', default=0, type=int)

# load the model, which we would like to evaluate
parser.add_argument('--size', default=18, type=int, help='resnet18 or wideresnet34')
parser.add_argument('--mode', default='baseline', type=str, help='baseline/margin')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')

## These parameters will be used to load models as well
parser.add_argument('--beta', default=5.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--alpha', default=0.0, type=float,
                    help='margin/margin_decor')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')

# pgd loss, we use this to compute weighted loss
parser.add_argument('--pgd_loss', default='margin', type=str)
parser.add_argument('--alpha_pgd', default=0.0, type=float,
                    help='margin')

parser.add_argument('--epsilon_eval', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num_steps_eval', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step_size_eval', default=0.0031, type=float,
                    help='perturb step size')

# specify random data distribution for evaluation
parser.add_argument('--random_init',default=True, help='initialize perturbations with random numbers')
  
args = parser.parse_args()
print(args)
if args.size == 18:
    model_size = 'resnet18'

if args.mode == "baseline":
    model_name = model_size + "_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_seed_' + str(args.seed) 

elif args.mode == "margin":
    model_name = model_size + "_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed) 

# settings
print("We are evaluating TRADES!!!")
model_dir = args.model_dir
load_dir = os.path.join(model_dir, args.mode)
load_dir = os.path.join(load_dir, model_name)

use_cuda = not args.no_cuda and torch.cuda.is_available()
seed = args.seed
device = torch.device('cuda:'+str(args.cuda))
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def main():
    # init model, Net() can be also used here for training
    if args.size == 18:
        torch.manual_seed(seed)
        model = ResNet18().to(device)

    model_name = os.path.join(load_dir, "last.checkpoint")
    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("we are evaluating: ", model_name)

    # white-box attack
    print('pgd white-box attack')
    acc_clean, acc_adv, acc_meaned_clean = eval_adv_test_whitebox(model, device, test_loader, args)
    print("pgd loss: {}, ACC_CLEAN: {:.2f}%, ACC_ADV: {:.2f}%".format(args.pgd_loss, acc_clean*100, acc_adv*100))
    print("acc clean traditional: {:.2f}%".format(acc_meaned_clean*100))


if __name__ == '__main__':
    main()