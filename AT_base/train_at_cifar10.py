import argparse
import json
from trainers import Trainer
from evaluators import Evaluator
# , EvaluatorAllAttacks

import utils
from utils import *
import os
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--cfg-path', default='configs/train.json', type=str,
                        help='path to config file')
    parser.add_argument('--data_root', default='../data', type=str,
                        help='path to dataset')
    parser.add_argument('--alg', default='adv_training', type=str,
                        help='Algorithm to train | Clean / Adv')
    parser.add_argument('--save_path', default='trained_models/cifar10/ADV_Train'', type=str,
                        help='path to save file')
    parser.add_argument('--attack_steps', default=10, type=int,
                        help='number of attack iterations (PGD-n)')
    parser.add_argument('--mode', default="baseline")
    parser.add_argument('--restore', default=None,
                        help='path to restore')

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--attack_eps', default=8, type=int)

    parser.add_argument('--dataset', default='CIFAR10', type=str)

    # model size
    parser.add_argument('--size', default=18, type=int)

    args = parser.parse_args()
    return args


def main(args):

    # Read configs
    with open(args.cfg_path, "r") as fp:
        configs = json.load(fp)

    # Update the configs based on command line args
    arg_dict = vars(args)
    for key in arg_dict:
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]
    
    configs = utils.ConfigMapper(configs)

    configs.attack_eps = float(configs.attack_eps) / 255
    configs.attack_lr = float(configs.attack_lr) / 255

    print("configs: ", configs.mode)

    configs.save_path = os.path.join(configs.save_path, configs.mode)
    experiment_name = exp_name(args)
    configs.save_path = os.path.join(configs.save_path, experiment_name)
    pathlib.Path(configs.save_path).mkdir(parents=True, exist_ok=True)

    # settings
    experiment_name = "resnet18_Adversarial_Training_margin" + '_lr_' + str(configs.lr) + '_alpha_' + str(configs.alpha) + '_seed_' + str(configs.seed) + '_epsilon_' + str(configs.attack_eps)

    trainer = Trainer(configs)
    trainer.train()


    print("training is over!!!")
if __name__ == '__main__':
    args = parse_args()
    main(args)

