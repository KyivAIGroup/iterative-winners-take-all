import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from nn.utils import random_choice, l0_sparsity
from mighty.utils.var_online import MeanOnline

__all__ = [
    "ParameterBinary",
    "PermanenceFixedSparsity",
    "PermanenceVaryingSparsity",
    "PermanenceVogels",
    "KWTAFunction",
    "KWTANet",
    "WTAInterface",
    "IterativeWTA",
    "IterativeWTASoft",
    "IterativeWTAVogels",
]


def normalize_presynaptic(mat):
    presum = mat.sum(dim=0, keepdim=True)
    presum += 1e-10
    mat /= presum


class ParameterBinary(nn.Parameter):
    def __new__(cls, data: torch.Tensor, learn=True, **kwargs):
        assert data.unique().tolist() == [0, 1]
        param = super().__new__(cls, data, requires_grad=False)
        param.learn = learn
        param.contribution = MeanOnline()
        return param

    @property
    def sparsity(self):
        return l0_sparsity(self.data)

    def reset(self):
        self.contribution.reset()

    def __repr__(self):
        shape = self.data.shape
        kind = "[learnable]" if self.learn else "[fixed]"
        return f"{self.__class__.__name__} {kind} {shape[0]} -> {shape[1]}"

    def update_contribution(self, freq_output: torch.Tensor):
        overlap = self.mean(dim=0) * freq_output
        self.contribution.update(overlap)

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        if not self.learn:
            # not learnable
            return
        for x, y in zip(x_pre, x_post):
            x = x.nonzero(as_tuple=True)[0]
            y = y.nonzero(as_tuple=True)[0]
            if len(x) == 0 or len(y) == 0:
                return
            if n_choose is None or n_choose >= len(x) * len(y):
                # full outer product
                self._do_update(x.unsqueeze(1), y, lr=lr)
            else:
                x = random_choice(x, n_choose=n_choose)
                y = random_choice(y, n_choose=n_choose)
                self._do_update(x, y, lr=lr)

    def _do_update(self, x_idx, y_idx, lr):
        self[x_idx, y_idx] = 1


class PermanenceFixedSparsity(ParameterBinary):

    def __new__(cls, data: torch.Tensor, learn=True, **kwargs):
        param = super().__new__(cls, data, learn=learn)
        permanence = torch.rand_like(data)
        normalize_presynaptic(permanence)
        param.permanence = permanence if learn else None
        return param

    def _do_update(self, x_idx, y_idx, lr):
        self.permanence[x_idx, y_idx] += lr

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        super().update(x_pre, x_post, n_choose=n_choose, lr=lr)
        self.normalize()

    def normalize(self):
        if not self.learn:
            return None
        normalize_presynaptic(self.permanence)
        k = math.ceil(self.count_nonzero() / self.size(1))
        topk = self.permanence.topk(k, dim=0)
        self.data.zero_()
        self.data[topk.indices, torch.arange(self.size(1)).unsqueeze(0)] = 1
        threshold = topk.values[-1].mean().item()
        return threshold


class PermanenceVaryingSparsity(PermanenceFixedSparsity):

    def __new__(cls, data: torch.Tensor, excitatory: bool, learn=True,
                output_sparsity_desired=(0.025, 0.1)):
        param = super().__new__(cls, data, learn=learn)
        param.excitatory = excitatory
        param.s_w = random.random() if learn else None
        param.output_sparsity_desired = output_sparsity_desired
        param.output_sparsity = MeanOnline()
        param.threshold = MeanOnline()
        param.permanences_removed = 0
        return param

    def reset(self):
        super().reset()
        self.output_sparsity.reset()
        self.threshold.reset()
        self.permanences_removed = 0

    def update_s_w(self, output_sparsity: float, gamma=0.1):
        s_inc = min(0.95, self.s_w * (1 + gamma))
        s_dec = max(0.05, self.s_w * (1 - gamma))
        s_w = self.s_w
        s_min, s_max = self.output_sparsity_desired
        if output_sparsity > s_max:
            s_w = s_dec if self.excitatory else s_inc
        elif output_sparsity < s_min:
            s_w = s_inc if self.excitatory else s_dec
        return s_w

    def update(self, x_pre, x_post, n_choose=1, lr=0.001):
        if not self.learn:
            # not learnable
            return
        self.output_sparsity.update(torch.Tensor([l0_sparsity(x_post)]))
        output_sparsity = self.output_sparsity.get_mean().item()
        self.s_w = self.update_s_w(output_sparsity)
        super().update(x_pre, x_post, n_choose=n_choose, lr=lr)

    def normalize(self):
        if not self.learn:
            return None
        normalize_presynaptic(self.permanence)
        k = math.ceil(self.s_w * self.size(0))
        topk = self.permanence.topk(k, dim=0)
        self.data.zero_()
        self.data[topk.indices, torch.arange(self.size(1)).unsqueeze(0)] = 1
        self.permanence *= self.data  # data serves as a mask
        threshold = topk.values[-1]
        self.permanences_removed += (self.permanence < threshold).sum().item()
        threshold = threshold.mean()
        self.threshold.update(threshold)
        return threshold.item()


class PermanenceVogels(PermanenceFixedSparsity):

    def update(self, x_pre, x_post, n_choose=1, lr=0.001,
               neighbors_coincident=1):
        assert len(x_pre) == len(x_post)
        window_size = (neighbors_coincident + 1)
        n_steps = len(x_pre)
        # multiply by ~10 to be on a similar learning time scale with
        # other learning rules
        lr_potentiation = 10 * lr / (n_steps * window_size)
        lr_depression = -10 * lr / (n_steps * (n_steps - window_size))
        for i, (x, y) in enumerate(zip(x_pre, x_post)):
            for j in range(max(0, i - neighbors_coincident), i + 1):
                # Potentiation
                x_recent = x_pre[j]
                alpha = lr_potentiation * (1 - (i - j) / window_size)
                ParameterBinary.update(self, x_pre=x_recent, x_post=y, lr=alpha)
            for j in range(0, i - neighbors_coincident - 1):
                # Depression
                x_past = x_pre[j]
                ParameterBinary.update(self, x_pre=x_past, x_post=y, lr=lr_depression)
        self.normalize()

    def normalize(self):
        self.permanence.clamp_min_(0)
        super().normalize()


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
            if isinstance(weight, ParameterBinary):
                weight.update(x_pre, x_post, n_choose=n_choose, lr=lr)

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
            if isinstance(param, PermanenceVaryingSparsity):
                thr = param.threshold.get_mean()
                if thr is not None:
                    thr = thr.item()
                thresholds[name] = thr
        return thresholds

    def weight_sparsity(self):
        sparsity = {name: param.sparsity
                    for name, param in self.named_parameters()
                    if isinstance(param, ParameterBinary)}
        return sparsity

    def s_w(self):
        nonzero_keep = {name: param.s_w
                        for name, param in self.named_parameters()
                        if isinstance(param, PermanenceVaryingSparsity)}
        return nonzero_keep

    def weight_contribution(self):
        contribution = {name: param.contribution.get_mean()
                        for name, param in self.named_parameters()
                        if isinstance(param, ParameterBinary)}
        return contribution

    def permanences_removed(self):
        removed = {name: param.permanences_removed
                   for name, param in self.named_parameters()
                   if isinstance(param, PermanenceVaryingSparsity)}
        return removed


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
        x = x.flatten(start_dim=1)
        y0 = x @ self.w_xy
        h = KWTAFunction.apply(x @ self.w_xh, self.kh)
        y = KWTAFunction.apply(y0 - h @ self.w_hy, ky)
        return h, y


class IterativeWTA(WTAInterface):
    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None):
        super().__init__(w_xy, w_xh, w_hy, w_yy=w_yy, w_hh=w_hh, w_yh=w_yh)
        self.freq = dict(
            y=MeanOnline(torch.zeros(w_xy.size(1), device=w_xy.device)),
            h=MeanOnline(torch.zeros(w_xh.size(1), device=w_xh.device))
        )

    def epoch_finished(self):
        super().epoch_finished()
        for freq in self.freq.values():
            freq.mean.fill_(0)
            freq.count = 0

    def forward(self, x):
        x = torch.atleast_2d(x)
        x = x.flatten(start_dim=1)
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0)
        y = torch.zeros_like(y0)
        t_start = int(max(h0.max().item(), y0.max().item()))
        # inh_h = torch.exp(-1 * self.freq['h'].get_mean())
        # inh_y = torch.exp(-1 * self.freq['y'].get_mean())
        inh_h = 1
        inh_y = 1
        for threshold in range(t_start, 0, -1):
            z_h = h0
            if self.w_hh is not None:
                z_h = z_h - h @ self.w_hh
            if self.w_yh is not None:
                z_h = z_h + y @ self.w_yh
            z_h = (z_h * inh_h) >= threshold

            z_y = y0 - h @ self.w_hy
            if self.w_yy is not None:
                z_y += y @ self.w_yy
            z_y = (z_y * inh_y) >= threshold

            h += z_h
            y += z_y
            h.clamp_max_(1)
            y.clamp_max_(1)

        self.freq['h'].update(h.mean(dim=0))
        self.freq['y'].update(y.mean(dim=0))

        return h, y


class IterativeWTAVogels(IterativeWTA):

    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None):
        super().__init__(w_xy, w_xh, w_hy, w_yy=w_yy, w_hh=w_hh, w_yh=w_yh)
        self.history = []

    def forward(self, x):
        self.history.clear()
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

        return h, y

    def update_weights(self, x, h, y, n_choose=1, lr=0.001):
        def _update_weight(weight, x_pre, x_post):
            if isinstance(weight, ParameterBinary):
                weight.update(x_pre, x_post, n_choose=n_choose, lr=lr)

        # Regular excitatory synapses update
        _update_weight(self.w_xy, x_pre=x, x_post=y)
        _update_weight(self.w_xh, x_pre=x, x_post=h)
        _update_weight(self.w_yy, x_pre=y, x_post=y)
        _update_weight(self.w_yh, x_pre=y, x_post=h)

        # Inhibitory synapses update
        assert isinstance(self.w_hh, PermanenceVogels)
        assert isinstance(self.w_hy, PermanenceVogels)
        z_h, z_y = zip(*self.history)
        self.w_hh.update(x_pre=z_h, x_post=z_h, n_choose=n_choose, lr=lr)
        self.w_hy.update(x_pre=z_h, x_post=z_y, n_choose=n_choose, lr=lr)
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
