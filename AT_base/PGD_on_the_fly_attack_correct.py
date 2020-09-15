import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from utils_pgd_correct import *

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def _pgd_whitebox(model,
                  X,
                  y,
                  args):
    # we use model attacker to compute forward pass
    # model attacker must be weighted-trained
    device = X.device
    print(device)
    # model_attacker is used to generate cheating test distribution
    out = model(X)
    acc = (out.data.max(1)[1] == y.data).float().mean()
    # print("mean acc: ", acc.item())
    X_pgd = Variable(X.data, requires_grad=True)
    batch_size = X.shape[0]

    if args.pgd_loss == "baseline":
        criterion_ce = nn.CrossEntropyLoss().to(device)
    else:
        criterion_ce = nn.CrossEntropyLoss(reduction='none').to(device)

    max_loss_pgd = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    for zz in range(10):
        if args.random_init:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon_eval, args.epsilon_eval).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        
        # ####################################################################
        # ########################### PGD LOSS ###############################
        # ####################################################################

        for _ in range(args.num_steps_eval):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
        
            # we use model_attacker to generate adversarial examples
            # so we use model_attacker to compute the loss
            perturbation = X_pgd - X

            # since we are generating adversarial examples using the learned distribution
            # this part should be the same as inner loop of trades.py
            with torch.enable_grad():
                outputs_adv = model(X_pgd)
                outputs = model(X)
                # loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                # loss = pgd_loss(args, outputs_adv, outputs, y, perturbation, criterion_kl, criterion_ce)
                loss_pgd, loss_individual_pgd = pgd_loss_ce(args, outputs_adv, outputs, y, perturbation, criterion_ce)
            
            loss_pgd.backward()
            eta = args.step_size_eval * X_pgd.grad.data.sign()
            
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -args.epsilon_eval, args.epsilon_eval)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            # break

        outputs_adv = model(X_pgd)
        label_mask = nn.functional.one_hot(y, 10).to(torch.bool)
        label_logit = outputs_adv[label_mask]
        others = outputs_adv[~label_mask].reshape(-1, 9)
        top_other_logit, _ = torch.max(others, dim=1)
        all_loss = (top_other_logit - label_logit).detach()

        # all_loss = F.cross_entropy(model(X_pgd), y, reduction='none').detach()

        delta = X_pgd - X
        max_delta[all_loss >= max_loss_pgd] = delta.detach()[all_loss >= max_loss_pgd]
        max_loss_pgd = torch.max(max_loss_pgd, all_loss)

        # break
    outputs_adv = model(X + max_delta)
    one_hot_adv = (outputs_adv.data.max(1)[1] == y.data).float().detach().cpu().numpy()

    label_mask = nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs_adv[label_mask]
    others = outputs_adv[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)
    loss_individual_pgd = (top_other_logit - label_logit).detach().cpu().numpy()
    # print("loss_individual_pgd shape: ", loss_individual_pgd.shape)
    # loss_individual_pgd = F.cross_entropy(outputs_adv, y, reduction='none').detach().cpu().numpy()

    outputs = model(X)
    one_hot = (outputs.data.max(1)[1] == y.data).float().detach().cpu().numpy()

    label_mask = nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs[label_mask]
    others = outputs[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)
    loss_individual = (top_other_logit - label_logit).detach().cpu().numpy()
    # print("loss_individual shape: ", loss_individual.shape)
    # loss_individual = F.cross_entropy(outputs, y, reduction='none').detach().cpu().numpy()
    
    perturbation = max_delta
    margin, weights_margin = compute_weights(args, outputs_adv, perturbation, y)

    acc = np.mean(one_hot)
    return margin, weights_margin, one_hot, one_hot_adv, acc, loss_individual, loss_individual_pgd

    # if args.random_init:
    #     random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon_eval, args.epsilon_eval).to(device)
    #     X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # ####################################################################
    # ########################### PGD LOSS ###############################
    # ####################################################################

    # for _ in range(args.num_steps_eval):
    #     opt = optim.SGD([X_pgd], lr=1e-3)
    #     opt.zero_grad()
        
    #     # we use model_attacker to generate adversarial examples
    #     # so we use model_attacker to compute the loss
    #     perturbation = X_pgd - X

    #     # since we are generating adversarial examples using the learned distribution
    #     # this part should be the same as inner loop of trades.py
    #     with torch.enable_grad():
    #         outputs_adv = model(X_pgd)
    #         outputs = model(X)
    #         # loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    #         # loss = pgd_loss(args, outputs_adv, outputs, y, perturbation, criterion_kl, criterion_ce)
    #         loss_pgd, loss_individual_pgd = pgd_loss_ce(args, outputs_adv, outputs, y, perturbation, criterion_ce)
        
    #     loss_pgd.backward()
    #     eta = args.step_size_eval * X_pgd.grad.data.sign()
        
    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    #     eta = torch.clamp(X_pgd.data - X.data, -args.epsilon_eval, args.epsilon_eval)
    #     X_pgd = Variable(X.data + eta, requires_grad=True)
    #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # outputs_adv = model(X_pgd)
    # one_hot_adv = (outputs_adv.data.max(1)[1] == y.data).float().detach().cpu().numpy()
    # loss_individual_pgd = F.cross_entropy(outputs_adv, y, reduction='none').detach().cpu().numpy()

    # outputs = model(X)
    # one_hot = (outputs.data.max(1)[1] == y.data).float().detach().cpu().numpy()
    # loss_individual = F.cross_entropy(outputs, y, reduction='none').detach().cpu().numpy()
    
    # # print("loss individual: ", loss_individual)
    # # print("loss individual: ", loss_individual_pgd)
    # perturbation = X_pgd - X
    # weights, weights_margin = compute_weights(args, outputs_adv, perturbation, y)

    # return weights, weights_margin, one_hot, one_hot_adv, acc.item(), loss_individual, loss_individual_pgd

def eval_adv_test_whitebox(model, device, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()

    # we save the weights from the model to be evaluated (model_base)
    # we compute the weighted accuracy

    # weight base is always unifrom
    margin_list = []
    weights_margin_list = []

    # we don't take mean to compute the acc, but one hot
    # so that we can compute the weighted accuracy
    one_hot_list = []
    one_hot_adv_list = []
    loss_individual_list = []
    loss_individual_adv_list = []

    acc_his = []
    count = 0
    
    for data, target in test_loader:
        # data, target = data.to(device), target.to(device)
        data, target = data.cuda(), target.cuda()
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        margin, weights_margin, one_hot, one_hot_adv, acc, loss_individual, loss_individual_adv = _pgd_whitebox(model, X, y, args)
        
        margin_list += list(margin)
        weights_margin_list += list(weights_margin)

        one_hot_list += list(one_hot)
        one_hot_adv_list += list(one_hot_adv)
        loss_individual_list += list(loss_individual)
        loss_individual_adv_list += list(loss_individual_adv)

        acc_his.append(acc)

        count += 1
        print("counter: ", count)
        # break

    margin_arr = np.array(margin_list)
    weights_margin_arr = np.array(weights_margin_list)

    print("one hot list len: ", len(one_hot_list))
    print("loss individual len: ", len(loss_individual_list))

    one_hot_arr = np.array(one_hot_list)
    one_hot_adv_arr = np.array(one_hot_adv_list)

    loss_individual_arr = np.array(loss_individual_list)
    loss_individual_adv_arr = np.array(loss_individual_adv_list)

    if args.mode == "baseline":
        model_name = "resnet18_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_seed_' + str(args.seed) 

    elif args.mode == "margin":
        model_name =  "resnet18_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed) 
    

    elif args.mode == "IAAT":
        model_name = "resnet18_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_seed_' + str(args.seed) 

    elif args.mode == "AT":
        model_name =  "resnet18_" + args.mode + '_lr_' + str(args.lr) + '_lambda_' + str(args.beta) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed) 

    save_weights_path = os.path.join('trained_weights_pgd_20_margin', args.mode)
    save_dir = os.path.join(save_weights_path, model_name)
    save_dir = os.path.join(save_dir, "epsilon_testing_" + str(args.epsilon_eval))
    # save_dir = os.path.join(save_dir, "alpha_testing_" + str(args.alpha_pgd))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'margin.npy'), margin_arr)
    # np.save(os.path.join(save_dir, 'weights_margin.npy'), weights_margin_arr)
    np.save(os.path.join(save_dir, 'one_hot.npy'), one_hot_arr)
    np.save(os.path.join(save_dir, 'one_hot_adv.npy'), one_hot_adv_arr)
    np.save(os.path.join(save_dir, 'loss.npy'), loss_individual_arr)
    np.save(os.path.join(save_dir, 'loss_adv.npy'), loss_individual_adv_arr)

    print("one hot arr ", one_hot_arr.shape)
    # weighted accuracy
    
    acc_clean = np.mean(one_hot_arr)
    acc_adv = np.sum(weights_margin_arr * one_hot_adv_arr) / np.sum(weights_margin_arr)

    acc_clean_meaned = np.mean(acc_his)

    return acc_clean, acc_adv, acc_clean_meaned