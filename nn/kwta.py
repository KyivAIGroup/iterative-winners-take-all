import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import random

from mighty.utils.signal import compute_sparsity
from nn.nn_utils import random_choice, l0_sparsity
from mighty.utils.var_online import MeanOnline

__all__ = [
    "ParameterWithPermanence",
    "ParameterBinary",
    "KWTAFunction",
    "KWTANet",
    "WTAInterface",
    "IterativeWTA",
    "IterativeWTASparse",
    "IterativeWTASoft",
    "IterativeWTAInhSTDP",
    "update_weights"
]


class ParameterBinary(nn.Parameter):
    SPARSITY_DESIRED = (0.025, 0.1)
    SPARSITY_TARGET = 0.05

    def __new__(cls, data: torch.Tensor, learn=True):
        dropout = random.uniform(0.05, 0.95)
        param = super().__new__(cls, data, requires_grad=False)
        assert data.unique().tolist() == [0, 1]
        param.learn = learn
        permanence = data.clone().float() * torch.rand_like(data)
        param.permanence = permanence if learn else None
        param.dropout = dropout if learn else None
        param.excitatory = None
        param.threshold = MeanOnline()
        param.sp_rmean = None  # output sparsity running mean
        return param

    def update_dropout2(self, sparsity: float, gamma=0.5, gamma_slow=0.1):
        if self.sp_rmean is None:
            self.sp_rmean = sparsity
        else:
            ds = sparsity - self.sp_rmean
            self.sp_rmean = gamma_slow * sparsity + (1 - gamma_slow) * self.sp_rmean
            # print(self.sp_rmean, ds, abs(ds) < 0.01)
            if abs(ds) < 0.01:
                return self.dropout
        ds = sparsity - self.SPARSITY_TARGET
        dropout_inc = ds * self.dropout
        if not self.excitatory:
            dropout_inc *= -1
        dropout = self.dropout + dropout_inc
        dropout = max(0.05, min(dropout, 0.95))
        dropout = dropout * gamma + (1 - gamma) * self.dropout
        if self.excitatory:
            dropout *= 0.99
            dropout = max(0.05, dropout)
        return dropout

    @property
    def sparsity(self):
        return l0_sparsity(self.data)

    def reset(self):
        self.threshold.reset()

    def update_dropout(self, output_sparsity: float, gamma=0.1):
        dropout_inc = gamma * 0.95 + (1 - gamma) * self.dropout
        dropout_dec = gamma * 0.05 + (1 - gamma) * self.dropout
        dropout = self.dropout
        s_min, s_max = self.SPARSITY_DESIRED
        if output_sparsity > s_max:
            dropout = dropout_inc if self.excitatory else dropout_dec
        elif output_sparsity < s_min:
            dropout = dropout_dec if self.excitatory else dropout_inc
        return dropout

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        if not self.learn:
            # not learnable
            return
        output_sparsity = l0_sparsity(x_post)
        self.dropout = self.update_dropout(output_sparsity)
        for x, y in zip(x_pre, x_post):
            x = x.nonzero(as_tuple=True)[0]
            y = y.nonzero(as_tuple=True)[0]
            if n_choose is None:
                # full outer product
                self.permanence[x.unsqueeze(1), y] += lr
            else:
                x = random_choice(x, n_choose=n_choose)
                y = random_choice(y, n_choose=n_choose)
                self.permanence[x, y] += lr
        self.normalize()

    def normalize(self):
        if not self.learn:
            return None
        self.permanence.clamp_min_(0)
        presum = self.permanence.sum(dim=0, keepdim=True)
        presum += 1e-10
        self.permanence /= presum
        # self.permanence /= self.permanence.max()  # only to avoid overflow
        perm = self.permanence.view(-1)
        perm = perm[perm.nonzero(as_tuple=True)[0]]
        pmax = perm.max()
        perm = perm[perm != pmax]  # the max element should survive
        n_active = len(perm)
        if n_active > 0:
            n_drop = max(1, int(self.dropout * n_active))
            # find k-th smallest value
            threshold = perm.kthvalue(n_drop).values.item()
        else:
            # pick any value in (0, pmax) range exclusively
            threshold = 0.5 * pmax
        self.permanence[self.permanence <= threshold] = 0
        self.data[:] = self.permanence > 0
        self.threshold.update(torch.Tensor([threshold]))
        return threshold

    def __repr__(self):
        shape = self.data.shape
        kind = "[learnable]" if self.learn else "[fixed]"
        return f"{self.__class__.__name__} {kind} {shape[0]} -> {shape[1]}"


class ParameterWithPermanence(ParameterBinary):

    def __new__(cls, permanence: torch.Tensor, sparsity: float, learn=True):
        if torch.cuda.is_available():
            permanence = permanence.cuda()
        n_active = math.ceil(permanence.nelement() * sparsity)
        presum = permanence.sum(dim=0, keepdim=True)
        permanence /= presum
        thr = permanence.view(-1).topk(n_active).values[-1]
        data = (permanence > thr).float()

        param = super().__new__(cls, data, learn=learn)
        param.permanence = permanence
        param.n_active = n_active
        return param

    def update(self, x_pre, x_post, lr=0.001):
        if not self.learn:
            # not learnable
            return
        # update permanence and data
        x_pre = torch.atleast_2d(x_pre)
        x_post = torch.atleast_2d(x_post)
        for x, y in zip(x_pre, x_post):
            self.permanence.addr_(x, y, alpha=lr)
        self.normalize()

    def normalize(self):
        if not self.learn:
            return None
        self.permanence.clamp_min_(0)
        presum = self.permanence.sum(dim=0, keepdim=True)
        presum += 1e-10
        self.permanence /= presum
        return super().normalize()
        perm = self.permanence.view(-1)
        threshold = perm.view(-1).topk(self.n_active).values[-1].item()
        self.data[:] = self.permanence > threshold
        return threshold


class KWTAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k):
        assert k is not None
        x = torch.atleast_2d(x)
        y = torch.zeros_like(x)
        if isinstance(k, int):
            winners = x.topk(k=k, dim=1, sorted=False).indices
            arange = torch.arange(x.shape[0], device=y.device).unsqueeze_(1)
            y[arange, winners] = 1
        else:
            assert x.shape[0] == len(k)
            for trial_id, xi in enumerate(x):
                winners = xi.topk(k=k[trial_id], sorted=False).indices
                y[trial_id, winners] = 1
        return y


class WTAInterface(nn.Module):
    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None):
        super().__init__()
        w_xy.excitatory = True
        w_xh.excitatory = True
        w_hy.excitatory = False
        if w_yy is not None:
            w_yy.excitatory = True
        if w_hh is not None:
            w_hh.excitatory = False
        if w_yh is not None:
            w_yh.excitatory = True
        self.w_xy = w_xy
        self.w_xh = w_xh
        self.w_hy = w_hy
        self.w_yy = w_yy
        self.w_hh = w_hh
        self.w_yh = w_yh

    def extra_repr(self) -> str:
        s = [f"{name}: {repr(param)}" for name, param in self.named_parameters()]
        return '\n'.join(s)

    def update_weights(self, x, h, y, n_choose=1, lr=0.001):
        def _update_weight(weight, x_pre, x_post):
            if isinstance(weight, ParameterWithPermanence):
                weight.update(x_pre, x_post, lr=lr)
            elif isinstance(weight, ParameterBinary):
                weight.update(x_pre, x_post, n_choose=n_choose)

        _update_weight(self.w_xy, x_pre=x, x_post=y)
        _update_weight(self.w_xh, x_pre=x, x_post=h)
        _update_weight(self.w_hy, x_pre=h, x_post=y)
        _update_weight(self.w_yy, x_pre=y, x_post=y)
        _update_weight(self.w_hh, x_pre=h, x_post=h)
        _update_weight(self.w_yh, x_pre=y, x_post=h)

    def epoch_finished(self):
        for param in self.parameters():
            param.reset()

    def kwta_thresholds(self):
        thresholds = {}
        for name, param in self.named_parameters():
            thr = param.threshold.get_mean()
            if thr is not None:
                thr = thr.item()
            thresholds[name] = thr
        return thresholds

    def weight_sparsity(self):
        sparsity = {name: param.sparsity
                    for name, param in self.named_parameters()}
        return sparsity

    def weight_dropout(self):
        dropout = {name: param.dropout
                   for name, param in self.named_parameters()}
        return dropout


class KWTANet(WTAInterface):
    def __init__(self, w_xy, w_xh, w_hy, kh=None, ky=None):
        # w_hh, w_yh, and w_yy are not defined for kWTA
        super().__init__(w_xy, w_xh, w_hy)
        assert kh is not None
        self.kh = kh
        self.ky = ky  # if None, must be specified in the forward pass

    def forward(self, x, ky=None):
        if ky is None:
            ky = self.ky
        x = torch.atleast_2d(x)
        y0 = x @ self.w_xy
        h = KWTAFunction.apply(x @ self.w_xh, self.kh)
        y = KWTAFunction.apply(y0 - h @ self.w_hy, ky)
        return h, y


class IterativeWTA(WTAInterface):

    def forward(self, x):
        x = torch.atleast_2d(x)
        x = x.flatten(start_dim=1)
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0)
        y = torch.zeros_like(y0)
        t_start = int(max(h0.max().item(), y0.max().item()))
        for threshold in range(t_start, 0, -1):
            z_h = h0
            if self.w_hh is not None:
                z_h = z_h - h @ self.w_hh
            if self.w_yh is not None:
                z_h = z_h + y @ self.w_yh
            z_h = z_h >= threshold

            z_y = y0 - h @ self.w_hy
            if self.w_yy is not None:
                z_y += y @ self.w_yy
            z_y = z_y >= threshold

            h += z_h
            y += z_y
            h.clamp_max_(1)
            y.clamp_max_(1)

        # TODO the same hack should be for 'h'
        empty_trials = ~(y.any(dim=1))
        if empty_trials.any():
            # This is particularly wrong because y != y_kwta even when k=1
            warnings.warn("iWTA resulted in a zero vector. "
                          "Activating one neuron manually.")
            h_kwta = KWTAFunction.apply(h0, 1)
            y_kwta = KWTAFunction.apply(y0 - h_kwta @ self.w_hy, 1)
            h[empty_trials] = h_kwta[empty_trials]
            y[empty_trials] = y_kwta[empty_trials]

        return h, y


class IterativeWTAInhSTDP(IterativeWTA):
    history = []
    N_COINCIDENT = 1

    def forward(self, x):
        x = torch.atleast_2d(x)
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0)
        y = torch.zeros_like(y0)
        t_start = int(max(h0.max().item(), y0.max().item()))
        for threshold in range(t_start, 0, -1):
            z_h = h0
            if self.w_hh is not None:
                z_h = z_h - h @ self.w_hh
            if self.w_yh is not None:
                z_h = z_h + y @ self.w_yh
            z_h = z_h >= threshold

            z_y = y0 - h @ self.w_hy
            if self.w_yy is not None:
                z_y += y @ self.w_yy
            z_y = z_y >= threshold

            h += z_h
            y += z_y
            h.clamp_max_(1)
            y.clamp_max_(1)

            self.history.append((z_h, z_y))

        # TODO the same hack should be for 'h'
        empty_trials = ~(y.any(dim=1))
        if empty_trials.any():
            # This is particularly wrong because y != y_kwta even when k=1
            warnings.warn("iWTA resulted in a zero vector. "
                          "Activating one neuron manually.")
            h_kwta = KWTAFunction.apply(h0, 1)
            y_kwta = KWTAFunction.apply(y0 - h_kwta @ self.w_hy, 1)
            h[empty_trials] = h_kwta[empty_trials]
            y[empty_trials] = y_kwta[empty_trials]

        return h, y

    def update_weights(self, x, h, y, n_choose=1, lr=0.001):
        def _update_weight(weight, x_pre, x_post):
            if isinstance(weight, ParameterWithPermanence):
                weight.update(x_pre, x_post, lr=lr)
            elif isinstance(weight, ParameterBinary):
                weight.update(x_pre, x_post, n_choose=n_choose)

        # Regular excitatory synapses update
        _update_weight(self.w_xy, x_pre=x, x_post=y)
        _update_weight(self.w_xh, x_pre=x, x_post=h)
        _update_weight(self.w_yy, x_pre=y, x_post=y)
        _update_weight(self.w_yh, x_pre=y, x_post=h)

        # Inhibitory synapses update
        assert isinstance(self.w_hh, ParameterWithPermanence)
        assert isinstance(self.w_hy, ParameterWithPermanence)
        n = (self.N_COINCIDENT + 1)
        nh = len(self.history)
        for i, (z_h, z_y) in enumerate(self.history):
            for j in range(max(0, i - self.N_COINCIDENT), i + 1):
                h_potentiation, _ = self.history[j]
                alpha = lr / (nh * n)
                self.w_hh.update(x_pre=h_potentiation, x_post=z_h, lr=alpha)
                self.w_hy.update(x_pre=h_potentiation, x_post=z_y, lr=alpha)
            for j in range(0, i - self.N_COINCIDENT - 1):
                h_depression, _ = self.history[j]
                alpha = -lr / (nh * (nh - n))
                self.w_hh.update(x_pre=h_depression, x_post=z_h, lr=alpha)
                self.w_hy.update(x_pre=h_depression, x_post=z_y, lr=alpha)

        self.history.clear()


class IterativeWTASoft(IterativeWTA):
    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None, hardness=1):
        super().__init__(w_xy, w_xh, w_hy, w_yy=w_yy, w_hh=w_hh, w_yh=w_yh)
        for p in self.parameters():
            delattr(p, 'permanence')
            p.data = p.data.float()
            p.requires_grad_(True)
        self.hardness = hardness

    def forward(self, x):
        x = torch.atleast_2d(x).float()
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        z_y = torch.zeros_like(y0)
        t_start = int(max(h0.max().item(), y0.max().item()))
        for threshold in range(t_start, 0, -1):
            z_h = h0
            if self.w_hh is not None:
                z_h = z_h - z_h @ self.w_hh
            if self.w_yh is not None:
                z_h = z_h + z_y @ self.w_yh
            z_h = (z_h >= threshold).float()

            z_y = y0 - z_h @ self.w_hy
            if self.w_yy is not None:
                z_y = z_y + z_y @ self.w_yy
            if self.training and threshold == 1:
                z_y = self.hardness * (z_y - threshold)
                z_y = F.hardsigmoid(z_y, inplace=True)
            else:
                z_y = (z_y >= threshold).float()

        return z_h, z_y


class IterativeWTASparse(IterativeWTA):

    def forward(self, x):
        x = torch.atleast_2d(x)
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0)
        y = torch.zeros_like(y0)
        t_start = int(max(h0.max().item(), y0.max().item()))
        z_h_prev = torch.zeros_like(h)
        z_y_prev = torch.zeros_like(y)
        for threshold in range(t_start, 0, -1):
            z_h = h0
            if self.w_hh is not None:
                z_h = z_h - z_h_prev @ self.w_hh
            if self.w_yh is not None:
                z_h = z_h + z_y_prev @ self.w_yh
            z_h = (z_h >= threshold).float()

            z_y = y0 - z_h @ self.w_hy
            if self.w_yy is not None:
                z_y += z_y_prev @ self.w_yy
            z_y = (z_y >= threshold).float()

            z_h_prev = z_h
            z_y_prev = z_y
            h += z_h
            y += z_y
            h.clamp_max_(1)
            y.clamp_max_(1)

        # TODO the same hack should be for 'h'
        empty_trials = ~(y.any(dim=1))
        if empty_trials.any():
            # This is particularly wrong because y != y_kwta even when k=1
            warnings.warn("iWTA resulted in a zero vector. "
                          "Activating one neuron manually.")
            h_kwta = KWTAFunction.apply(h0, 1)
            y_kwta = KWTAFunction.apply(y0 - h_kwta @ self.w_hy, 1)
            h[empty_trials] = h_kwta[empty_trials]
            y[empty_trials] = y_kwta[empty_trials]

        return h, y


def update_weights(w, x_pre, x_post, n_choose=1):
    if x_pre.ndim == 2:
        for x, y in zip(x_pre, x_post):
            update_weights(w, x_pre=x, x_post=y, n_choose=n_choose)
        return

    x_pre_idx = x_pre.nonzero(as_tuple=True)[0]
    if len(x_pre_idx) == 0:
        warnings.warn("'x_pre' is a zero vector")
        return
    x_post_idx = x_post.nonzero(as_tuple=True)[0]
    if len(x_post_idx) == 0:
        warnings.warn("'x_post' is a zero vector")
        return
    if n_choose is None:
        # update all combinations
        w[x_pre_idx.unsqueeze(1), x_post_idx] = 1
    else:
        idx_pre = random_choice(x_pre_idx, n_choose=n_choose)
        idx_post = random_choice(x_post_idx, n_choose=n_choose)
        w[idx_pre, idx_post] = 1
