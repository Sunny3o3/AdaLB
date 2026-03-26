import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class AdaLB(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, gamma=0, lr_decay=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            lr_decay=lr_decay
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr_decay', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaLB does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['B_old'] = 0
                    state['B_new'] = 1

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                beta1, beta2_default = group['betas']
                beta2 = state['B_old'] / state['B_new']
                gamma = group['gamma']
                lr_decay = group['lr_decay']

                state['step'] += 1
                step = state['step']

                state['B_old'] += math.pow(gamma, step)
                state['B_new'] += math.pow(gamma, step + 1)

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad - exp_avg, grad - exp_avg)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1

                if lr_decay:
                    step_size = step_size / math.sqrt(state['step'])

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

    def denominator(self):
        denom = np.array([0])
        denom_sum = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaLB does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq_np = exp_avg_sq.detach().cpu().numpy().reshape(-1)
                denom = np.concatenate((denom, exp_avg_sq_np))
                denom_sum += torch.sum(exp_avg_sq)

        return denom, denom_sum