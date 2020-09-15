import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_loss_ce(args, outputs_adv, outputs, y, perturbation, criterion_ce):
    batch_size = outputs.shape[0]
    device = outputs_adv.device

    if args.pgd_loss == 'baseline':
        loss_ce = criterion_ce(outputs_adv, y)
        
    elif args.pgd_loss == "margin":
        # we compute the margin for adversarial examples

        """
        compute margin weights
        """
        proba = F.softmax(outputs_adv, dim=-1)
        proba_y = proba[list(range(batch_size)),y]
        proba_tops, ids_tops = torch.topk(proba,2)

        proba_top1 = proba_tops[:,0]
        proba_top2 = proba_tops[:,1]

        ids_top1 = ids_tops[:,0]
        ids_top2 = ids_tops[:,1]

        top1_t = (ids_top1.long() != y).float() * proba_top1
        top2_t = (ids_top1.long() == y).float() * proba_top2

        proba_t = top1_t + top2_t
        margin = proba_y - proba_t
        exp_coe = torch.exp(-margin * args.alpha_pgd)

        """
        cross entropy, if enabled
        """
        loss_individual = criterion_ce(outputs_adv, y)

        """
        margin loss, if enabled
        """
        # label_mask = nn.functional.one_hot(y, 10).to(torch.bool)
        # label_logit = outputs_adv[label_mask]
        # others = outputs_adv[~label_mask].reshape(-1, 9)
        # top_other_logit, _ = torch.max(others, dim=1)

        # loss_individual = top_other_logit - label_logit

        """
        unweighted loss, if enabled
        """
        # loss_ce = torch.sum(loss_individual)

        """
        weighted loss, if enabled
        """
        loss_ce = torch.sum(loss_individual * exp_coe)

    return loss_ce, loss_individual

def compute_weights(args,outputs_adv, perturbation, y):
    batch_size = outputs_adv.shape[0]

    if args.pgd_loss == 'baseline': 
        weights = np.ones(batch_size) / batch_size
        weights_margin = weights

    elif args.pgd_loss == "margin":
        # margin-based gaussian
        # we compute the margin for adversarial examples
        proba = F.softmax(outputs_adv, dim=-1)

        proba_y = proba[list(range(batch_size)),y]
        proba_tops, ids_tops = torch.topk(proba,2)

        proba_top1 = proba_tops[:,0]
        proba_top2 = proba_tops[:,1]

        ids_top1 = ids_tops[:,0]
        ids_top2 = ids_tops[:,1]

        top1_t = (ids_top1.long() != y).float() * proba_top1
        top2_t = (ids_top1.long() == y).float() * proba_top2

        proba_t = top1_t + top2_t
        margin = proba_y - proba_t
        exp_coe = torch.exp(-margin * args.alpha_pgd)

        # WEIGHTS
        margin = -margin.detach().cpu().numpy()
        weights_margin = exp_coe.detach().cpu().numpy()
         
    return margin, weights_margin