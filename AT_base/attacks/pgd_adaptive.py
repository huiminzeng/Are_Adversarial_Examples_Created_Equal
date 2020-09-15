## Code for PGD attacker
import torch
import torch.nn.functional as F


class PGDAttackerAdaptive(object):

    def __init__(self):
        pass

    def attack(self, x, y, net, attack_steps, attack_lr_arr, attack_eps_arr, random_init=True, target=None,
               clamp=(0, 1)):
        """
        :param x: Inputs to perturb
        :param y: Corresponding ground-truth labels
        :param net: Network to attack
        :param attack_steps: Number of attack iterations
        :param attack_lr_arr: Learning rate of attacker (should be an array of batch size)
        :param attack_eps_arr: Attack epsilon of attacker (should be an array of batch size)
        :param random_init: If true, uses random initialization
        :param target: If not None, attacks to the chosen class. Dimension of target should be same as labels
        :param clamp: Clamping values
        :return:
        """

        x_adv = x.clone()

        assert x.size(0) == attack_eps_arr.size(0)
        assert x.size(0) == attack_lr_arr.size(0)

        attack_lr_arr = attack_lr_arr.view(attack_lr_arr.size(0), 1, 1, 1)
        attack_eps_arr = attack_eps_arr.view(attack_eps_arr.size(0), 1, 1, 1)

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps_arr

        for i in range(attack_steps):
            x_adv.requires_grad = True

            net.zero_grad()
            logits = net(x_adv)

            if target is None:
                # Untargeted attacks - gradient ascent
                loss = F.cross_entropy(logits, y)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv + attack_lr_arr * grad

            else:
                # Targeted attacks - gradient descent
                assert target.size() == y.size()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv - attack_lr_arr * grad

            # Projection
            diff = x_adv - x
            diff = torch.max(diff, -attack_eps_arr)
            diff = torch.min(diff, attack_eps_arr)
            x_adv = diff + x
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *clamp)

        return x_adv






