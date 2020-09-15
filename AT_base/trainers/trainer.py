import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import models
import torch.optim as optim
import os.path as osp
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import utils
from attacks import PGDAttacker

class Trainer:
    def __init__(self, args):

        self.args = args

        # Creating data loaders
        kwargs = {'num_workers': 4, 'pin_memory': True}
        if args.dataset == 'MNIST':
            # setup data loader
            self.train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                            transform=T.ToTensor()),
                batch_size=args.batch_size, shuffle=True, **kwargs)

            self.val_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False,
                            transform=T.ToTensor()),
                            batch_size=args.batch_size, shuffle=False, **kwargs)
            
            # initialize model
            torch.manual_seed(args.seed)
            self.model = models.SmallCNN()

        elif args.dataset == 'CIFAR10':
            transform_train = T.Compose([
                                T.Pad(4, padding_mode='reflect'),
                                T.RandomCrop(32),
                                T.RandomHorizontalFlip(),
                                T.ToTensor()
                            ])
            transform_test = T.Compose([
                                T.ToTensor()
                            ])

            self.train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_root, train=True, download=True,
                                transform=transform_train),
                                batch_size=args.batch_size, shuffle=True, **kwargs)
            self.val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_root, train=False, transform=transform_test),
                                batch_size=args.batch_size, shuffle=True, **kwargs)
            # initialize model
            torch.manual_seed(args.seed)
            self.model = models.ResNet18()
        
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.optimizer = optim.SGD(self.model.parameters(), args.lr,
                                   momentum=0.9, weight_decay=args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)

        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        self.save_path = args.save_path
        self.epoch = 0

        # resume from checkpoint

        ckpt_path = osp.join(self.save_path, 'checkpoint.pth')
        if osp.exists(ckpt_path):
            self._load_from_checkpoint(ckpt_path)
        elif args.restore:
            self._load_from_checkpoint(args.restore)

        cudnn.benchmark = True
        self.attacker = PGDAttacker(args.attack_eps)

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, ckpt_path):
        print('Loading model from {} ...'.format(ckpt_path))
        model_data = torch.load(ckpt_path)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.lr_scheduler.load_state_dict(model_data['lr_scheduler'])
        self.epoch = model_data['epoch'] + 1
        print('Model loaded successfully')

    def _save_checkpoint(self):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['lr_scheduler'] = self.lr_scheduler.state_dict()
        model_data['epoch'] = self.epoch
        torch.save(model_data, osp.join(self.save_path, 'checkpoint.pth'))
        print("Saving to: ", self.save_path)

    def train(self):

        losses = utils.AverageMeter()

        while self.epoch < self.args.nepochs:
            self.model.train()
            correct = 0
            total = 0
            start_time = time.time()

            for i, data in enumerate(self.train_loader):
                input, target = data
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

                if self.args.alg == 'adv_training':
                    input = self.attacker.attack(input, target, self.model, self.args.attack_steps, self.args.attack_lr,
                                                 random_init=True, target=None, alpha=self.args.alpha)

                # compute output
                self.optimizer.zero_grad()
                logits = self.model(input)

                if self.args.mode == 'margin':
                    proba = F.softmax(logits, dim=-1)
                    proba_y = proba[list(range(input.size(0))),target]
                    proba_tops, ids_tops = torch.topk(proba,2)

                    proba_top1 = proba_tops[:,0]
                    proba_top2 = proba_tops[:,1]

                    ids_top1 = ids_tops[:,0]
                    ids_top2 = ids_tops[:,1]

                    top1_t = (ids_top1.long() != target).float() * proba_top1
                    top2_t = (ids_top1.long() == target).float() * proba_top2

                    proba_t = top1_t + top2_t
                    margin = proba_y - proba_t
                    exp_coe = torch.exp(-margin * self.args.alpha)

                    loss = F.cross_entropy(logits, target, reduction='none')
                    loss = torch.mean(loss * exp_coe)
                
                elif self.args.mode == 'baseline':
                    loss = F.cross_entropy(logits, target)

                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(logits, dim=1)
                correct += (pred == target).sum()
                total += target.size(0)

                # measure accuracy and record loss
                losses.update(loss.data.item(), input.size(0))
                # break

            self.epoch += 1
            self.lr_scheduler.step()
            end_time = time.time()
            batch_time = end_time - start_time

            acc = (float(correct) / total) * 100
            message = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'.format(self.epoch, batch_time, loss.item(), acc)
            self._log(message)
            self._save_checkpoint()

            # Evaluation
            nat_acc = self.eval()
            adv_acc = self.eval_adversarial()
            self._log('Natural accuracy: {}'.format(nat_acc))
            self._log('Adv accuracy: {}'.format(adv_acc))


    def eval(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)
            # break

        accuracy = (float(correct) / total) * 100
        return accuracy

    def eval_adversarial(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            input = self.attacker.attack(input, target, self.model, self.args.attack_steps, self.args.attack_lr,
                                         random_init=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)
            # break

        accuracy = (float(correct) / total) * 100
        return accuracy



