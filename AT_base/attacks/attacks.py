import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import foolbox
from art.attacks.evasion import carlini
from art.classifiers.classifier import Classifier
from art.classifiers import PyTorchClassifier


class Attackers(object):

    def __init__(self, net, num_classes, attack_type, eps, batch_size):
        self.num_classes = num_classes
        self.net = net
        self.fmodel = foolbox.models.PyTorchModel(net, bounds=(0, 1), num_classes=10)
        
        if attack_type == "DeepFool":
            attacker = foolbox.attacks.DeepFoolLinfinityAttack(self.fmodel)
        elif attack_type == "MIFGSM":
            attacker = foolbox.attacks.MomentumIterativeAttack(self.fmodel, distance=foolbox.distances.Linfinity)
        elif attack_type == "CW":
            a0 = 2*eps/20
            classifier = PyTorchClassifier(clip_values= (0,1), model=net, 
                                           loss=None, optimizer=None, input_shape=(batch_size, 3, 32, 32), 
                                           nb_classes=num_classes, channel_index=3, defences=None, preprocessing=(0, 1))        
            attacker = carlini.CarliniLInfMethod(classifier, learning_rate=a0/2, targeted=False,
                                           max_iter=40, eps=eps, batch_size=128)
            
            
        self.attacker = attacker
        self.attack_type = attack_type
        self.eps = eps

    def attack(self, x, y):
        x = x.cpu()
        y = y.cpu()
        if self.attack_type == "CW":
            advdata = self.attacker.generate(x.numpy(), kwargs_1=y.numpy())
            advdata = torch.from_numpy(advdata)
            return advdata.view(-1, 3, 32, 32).cuda()
        elif self.attack_type == "MIFGSM":
            advdata = self.attacker(x.numpy(), y.numpy(), binary_search=False, 
                               epsilon=self.eps, stepsize=self.eps/40.001, iterations=40, return_early=False)
        else:
            advdata = self.attacker(x.numpy(), y.numpy())
        
        if type(advdata)==type(None):
            return None            
        else:
            advdata = torch.from_numpy(advdata)
            
            if (advdata-x).abs().max() <= self.eps:
                return advdata.view(-1, 3, 32, 32).cuda()
            else:
                return None    
                
