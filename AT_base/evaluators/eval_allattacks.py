import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import models
import os.path as osp
import torch.backends.cudnn as cudnn
from attacks import PGDAttacker, Attackers
import numpy as np
import torchvision.utils as vutils

class EvaluatorAllAttacks:
    def __init__(self, args):

        self.args = args

        transformer = T.Compose([
            T.ToTensor()
        ])
        kwargs = {'num_workers': 4, 'pin_memory': True}

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, transform=transformer),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # Create model, optimizer and scheduler
        self.model = models.WRN(depth=32, width=10, num_classes=10)
        self.model = torch.nn.DataParallel(self.model).cuda()

        # Loading model
        assert self.args.restore is not None

        model_data = torch.load(self.args.restore)
        self.model.load_state_dict(model_data['model'])
        self.model.eval()

        cudnn.benchmark = True
        self.save_path = self.args.save_path

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def eval(self):
        # attack_list = ['PGD', 'DeepFool', 'MIFGSM', 'LBFGS']
        attack_list = ['CW']
        for attack in attack_list:
            acc = self.eval_worker(attack)
            message = 'Attack: {}, Acc:{}'.format(attack, acc)
            self._log(message)
        return acc

    def eval_worker(self, attack):
        correct = 0
        total = 0
        err = 0
        print('Evaluating {}'.format(attack))
        
        attack_eps = 8.0/255
        if attack == 'PGD':
            attacker = PGDAttacker(attack_eps)
        else:
            attacker = Attackers(self.model, 10, attack, attack_eps, self.args.batch_size)
        
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            if attack == 'PGD':
                attack_steps = 10
                attack_lr = 2.0 / 255
                pert = attacker.attack(input, target, self.model, attack_steps, attack_lr,
                                        random_init=True)
                with torch.no_grad():
                    output = self.model(pert)
                
                _, pred = torch.max(output, dim=1)
                correct += (pred == target).sum()
                total += target.size(0)
            elif attack == 'CW':
                pert = attacker.attack(input, target)
                with torch.no_grad():
                    output = self.model(pert)
                
                _, pred = torch.max(output, dim=1)
                correct += (pred == target).sum()
                total += target.size(0)
            else:
                for j in range(input.size(0)):
                    inp = input[j].unsqueeze(0)
                    tar = target[j].unsqueeze(0)
                    att = attacker.attack(inp, tar)
                    if att is not None:
                        with torch.no_grad():
                            output = self.model(att)
                            
                        _, pred = torch.max(output, dim=1)
                        if pred.item() != tar.item():
                            err += 1
                    total += 1
        print('Evaluated on {} samples'.format(total))
                            
        if attack == 'PGD' or attack == 'CW':
            accuracy = (float(correct) / total) * 100
        else:
            accuracy = (1.0 - float(err) / total) * 100
        
        return accuracy
