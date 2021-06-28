import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from mighty.utils.signal import compute_sparsity
from nn.nn_utils import random_choice

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

LEARNING_RATE = 0.001


class ParameterBinary(nn.Parameter):
    def __new__(cls, data, learn=True):
        param = super().__new__(cls, data, requires_grad=False)
        param.learn = learn
        return param

    def update(self, x_pre, x_post, n_choose=1):
        if not self.learn:
            # not learnable
            return
        update_weights(self.data, x_pre=x_pre, x_post=x_post,
                       n_choose=n_choose)

    def __repr__(self):
        shape = self.data.shape
        kind = "[learnable]" if self.learn else "[fixed]"
        return f"{self.__class__.__name__} {kind} {shape[0]} -> {shape[1]}"


class ParameterWithPermanence(ParameterBinary):

    def __new__(cls, permanence: torch.Tensor, sparsity: float, learn=True):
        n_active = math.ceil(permanence.nelement() * sparsity)
        presum = permanence.sum(dim=0, keepdim=True)
        permanence /= presum
        data = torch.zeros_like(permanence, dtype=torch.int32)
        data.view(-1)[permanence.view(-1).topk(n_active).indices] = 1

        param = super().__new__(cls, data, learn=learn)
        param.permanence = permanence
        param.n_active = n_active
        return param

    @property
    def sparsity(self):
        return self.n_active / self.data.nelement()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.permanence.clone(memory_format=torch.preserve_format), self.sparsity, self.learn)
            memo[id(self)] = result
            return result

    def update(self, x_pre, x_post, alpha=LEARNING_RATE):
        if not self.learn:
            # not learnable
            return
        # update permanence and data
        x_pre = torch.atleast_2d(x_pre)
        x_post = torch.atleast_2d(x_post)
        for x, y in zip(x_pre, x_post):
            self.permanence.addr_(x, y, alpha=alpha)
        self.normalize()

    def normalize(self):
        presum = self.permanence.sum(dim=0, keepdim=True)
        self.permanence /= presum
        perm = self.permanence.view(-1)
        self.data.zero_()
        self.data.view(-1)[perm.topk(self.n_active).indices] = 1


class KWTAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k):
        assert k is not None
        x = torch.atleast_2d(x)
        y = torch.zeros_like(x, dtype=torch.int32)
        if isinstance(k, int):
            winners = x.topk(k=k, dim=1, sorted=False).indices
            y[torch.arange(x.shape[0], device=y.device).unsqueeze_(1),
              winners] = 1
        else:
            assert x.shape[0] == len(k)
            for trial_id, xi in enumerate(x):
                winners = xi.topk(k=k[trial_id], sorted=False).indices
                y[trial_id, winners] = 1
        return y


class WTAInterface(nn.Module):
    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None):
        super().__init__()
        self.w_xy = w_xy
        self.w_xh = w_xh
        self.w_hy = w_hy
        self.w_yy = w_yy
        self.w_hh = w_hh
        self.w_yh = w_yh
        self.monitor = None

    def extra_repr(self) -> str:
        s = [f"{name}: {repr(param)}" for name, param in self.named_parameters()]
        return '\n'.join(s)

    def set_monitor(self, monitor):
        self.monitor = monitor

    def update_weights(self, x, h, y, n_choose=1):
        def _update_weight(weight, x_pre, x_post):
            if isinstance(weight, ParameterWithPermanence):
                weight.update(x_pre, x_post)
            elif isinstance(weight, ParameterBinary):
                weight.update(x_pre, x_post, n_choose=n_choose)

        _update_weight(self.w_xy, x_pre=x, x_post=y)
        _update_weight(self.w_xh, x_pre=x, x_post=h)
        _update_weight(self.w_hy, x_pre=h, x_post=y)
        _update_weight(self.w_yy, x_pre=y, x_post=y)
        _update_weight(self.w_hh, x_pre=h, x_post=h)
        _update_weight(self.w_yh, x_pre=y, x_post=h)

    def weight_sparsity(self):
        sparsity = {name: compute_sparsity(param.data.float())
                    for name, param in self.named_parameters()}
        return sparsity


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
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0, dtype=torch.int32)
        y = torch.zeros_like(y0, dtype=torch.int32)
        t_start = max(h0.max().item(), y0.max().item())
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

            h |= z_h
            y |= z_y

            if self.monitor is not None:
                self.monitor.iwta_iteration(z_h, z_y)

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
        h = torch.zeros_like(h0, dtype=torch.int32)
        y = torch.zeros_like(y0, dtype=torch.int32)
        t_start = max(h0.max().item(), y0.max().item())
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

            h |= z_h
            y |= z_y

            if self.monitor is not None:
                self.monitor.iwta_iteration(z_h, z_y)

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

    def update_weights(self, x, h, y, n_choose=1):
        def _update_weight(weight, x_pre, x_post):
            if isinstance(weight, ParameterWithPermanence):
                weight.update(x_pre, x_post)
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
        for i, (z_h, z_y) in enumerate(self.history):
            for j in range(0, i - self.N_COINCIDENT):
                h_depression, _ = self.history[j]
                alpha = -0.2 * LEARNING_RATE / len(self.history)
                self.w_hh.update(x_pre=h_depression, x_post=z_h, alpha=alpha)
                self.w_hy.update(x_pre=h_depression, x_post=z_y, alpha=alpha)
            for j in range(max(0, i - self.N_COINCIDENT), i + 1):
                h_coinc, _ = self.history[j]
                alpha = LEARNING_RATE / len(self.history)
                self.w_hh.update(x_pre=h_coinc, x_post=z_h, alpha=alpha)
                self.w_hy.update(x_pre=h_coinc, x_post=z_y, alpha=alpha)

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

            if self.monitor is not None:
                self.monitor.iwta_iteration(z_h, z_y)

        return z_h, z_y


class IterativeWTASparse(IterativeWTA):

    def forward(self, x):
        x = torch.atleast_2d(x)
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0, dtype=torch.int32)
        y = torch.zeros_like(y0, dtype=torch.int32)
        t_start = max(h0.max().item(), y0.max().item())
        z_h_prev = torch.zeros_like(h)
        z_y_prev = torch.zeros_like(y)
        for threshold in range(t_start, 0, -1):
            z_h = h0
            if self.w_hh is not None:
                z_h = z_h - z_h_prev @ self.w_hh
            if self.w_yh is not None:
                z_h = z_h + z_y_prev @ self.w_yh
            z_h = (z_h >= threshold).int()

            z_y = y0 - z_h @ self.w_hy
            if self.w_yy is not None:
                z_y += z_y_prev @ self.w_yy
            z_y = (z_y >= threshold).int()

            z_h_prev = z_h
            z_y_prev = z_y
            h |= z_h
            y |= z_y

            if self.monitor is not None:
                self.monitor.iwta_iteration(z_h, z_y, id_=1)

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
