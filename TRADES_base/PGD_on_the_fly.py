import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from utils import *


def _pgd_whitebox(model,
                  X,
                  y,
                  args):

    device = X.device
    out = model(X)
    acc = (out.data.max(1)[1] == y.data).float().mean()
    
    X_pgd = Variable(X.data, requires_grad=True)
    batch_size = X.shape[0]

    if args.random_init:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon_eval, args.epsilon_eval).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(args.num_steps_eval):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        eta = args.step_size_eval * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -args.epsilon_eval, args.epsilon_eval)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        break

    acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().mean()

    return acc.item(), acc_pgd.item()

def eval_adv_test_whitebox(model, device, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_acc_total = []
    natural_acc_total = []
    test_batch_num = len(test_loader)
    counter = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        acc_natural, acc_robust = _pgd_whitebox(model, X, y, args)
        robust_acc_total.append(acc_robust)
        natural_acc_total.append(acc_natural)
        counter += 1
        break
    return np.mean(natural_acc_total), np.mean(robust_acc_total)