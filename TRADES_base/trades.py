import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                args,
                distance='l_inf'):
    device = x_natural.device
    # define KL-loss
    if args.mode == "baseline":
        criterion_kl = nn.KLDivLoss(size_average=False).to(device)
        criterion_ce = nn.CrossEntropyLoss().to(device)
    else:
        criterion_kl = nn.KLDivLoss(reduction='none').to(device)
        criterion_ce = nn.CrossEntropyLoss(reduction='none').to(device)

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(args.num_steps):
            perturbation = x_adv - x_natural
            x_adv.requires_grad_()
            with torch.enable_grad():
                outputs = model(x_natural)
                outputs_adv = model(x_adv)
                ####################################################################
                ############################# MODE #################################
                ####################################################################
                if args.mode == 'baseline':
                    loss_kl = criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs, dim=1))

                elif args.mode == "margin":
                    # margin-based gaussian
                    # we compute the margin for adversarial examples
                    proba = F.softmax(outputs_adv, dim=-1)
                    proba_y = proba[list(range(x_natural.size(0))),y]
                    proba_tops, ids_tops = torch.topk(proba,2)

                    proba_top1 = proba_tops[:,0]
                    proba_top2 = proba_tops[:,1]

                    ids_top1 = ids_tops[:,0]
                    ids_top2 = ids_tops[:,1]

                    top1_t = (ids_top1.long() != y).float() * proba_top1
                    top2_t = (ids_top1.long() == y).float() * proba_top2

                    proba_t = top1_t + top2_t
                    margin = proba_y - proba_t
                    exp_coe = torch.exp(-margin * args.alpha)
                    
                    loss_kl = criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs, dim=1))
                    loss_kl = torch.sum(loss_kl, dim=-1)
                    loss_kl = torch.sum(loss_kl * exp_coe)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    outputs = model(x_natural)
    outputs_adv = model(x_adv)
    # Take mean directly, regardless reduction
    loss_natural = torch.mean(criterion_ce(outputs, y))

    ####################################################################
    ############################# MODE #################################
    ####################################################################
    if args.mode == 'baseline':
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs, dim=1))

    elif args.mode == "margin":
        # margin-based gaussian
        # we compute the margin for adversarial examples
        proba = F.softmax(outputs_adv, dim=-1)
        proba_y = proba[list(range(x_natural.size(0))),y]
        proba_tops, ids_tops = torch.topk(proba,2)

        proba_top1 = proba_tops[:,0]
        proba_top2 = proba_tops[:,1]

        ids_top1 = ids_tops[:,0]
        ids_top2 = ids_tops[:,1]

        top1_t = (ids_top1.long() != y).float() * proba_top1
        top2_t = (ids_top1.long() == y).float() * proba_top2

        proba_t = top1_t + top2_t
        margin = proba_y - proba_t
        exp_coe = torch.exp(-margin * args.alpha)

                    
        loss_robust = criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs, dim=1))
        loss_robust = torch.sum(loss_robust, dim=-1)

        if args.normalize:
            loss_robust = torch.sum(loss_robust * exp_coe)
            loss_robust /= torch.sum(exp_coe).detach()
        else:
            loss_robust = torch.mean(loss_robust * exp_coe)
        
    loss = loss_natural + args.beta * loss_robust

    return loss, loss_robust