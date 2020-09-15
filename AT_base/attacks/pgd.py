## Code for PGD attacker
import torch
import torch.nn.functional as F


class PGDAttacker(object):

    def __init__(self, attack_eps):
        self.attack_eps = attack_eps

    def attack(self, x, y, net, attack_steps, attack_lr, random_init=True, target=None, clamp=(0, 1), alpha=0):
        """
        :param x: Inputs to perturb
        :param y: Corresponding ground-truth labels
        :param net: Network to attack
        :param attack_steps: Number of attack iterations
        :param attack_lr: Learning rate of attacker
        :param random_init: If true, uses random initialization
        :param target: If not None, attacks to the chosen class. Dimension of target should be same as labels
        :return:
        """

        x_adv = x.clone()

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.attack_eps

        for i in range(attack_steps):
            x_adv.requires_grad = True

            net.zero_grad()
            logits = net(x_adv)

            if target is None:
                # Untargeted attacks - gradient ascent
                proba = F.softmax(logits, dim=-1)
                proba_y = proba[list(range(x_adv.size(0))),y]
                proba_tops, ids_tops = torch.topk(proba,2)

                proba_top1 = proba_tops[:,0]
                proba_top2 = proba_tops[:,1]

                ids_top1 = ids_tops[:,0]
                ids_top2 = ids_tops[:,1]

                top1_t = (ids_top1.long() != y).float() * proba_top1
                top2_t = (ids_top1.long() == y).float() * proba_top2

                proba_t = top1_t + top2_t
                margin = proba_y - proba_t
                exp_coe = torch.exp(-margin * alpha)

                loss = F.cross_entropy(logits, y, reduction='none')

                loss = torch.mean(exp_coe * loss)
                # print("pgd loss: ", loss.item())
                # print("alpha: ", alpha)
                # print("exp: ", exp_coe[:20])

                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv + attack_lr * grad
                # print("attack lr: ", attack_lr)
                # print("attack eps: ", self.attack_eps)

            else:
                # Targeted attacks - gradient descent
                assert target.size() == y.size()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv - attack_lr * grad

            # Projection
            # print("epsilon: ", self.attack_eps)
            x_adv = x + torch.clamp(x_adv - x, min=-self.attack_eps, max=self.attack_eps)

            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *clamp)

        return x_adv






