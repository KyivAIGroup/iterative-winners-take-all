import warnings
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np


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


class KWTANet(nn.Module):
    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None, kh=None, ky=None):
        super().__init__()
        self.w_xy = w_xy
        self.w_xh = w_xh
        self.w_hy = w_hy
        self.w_yy = w_yy
        self.w_hh = w_hh
        self.w_yh = w_yh
        assert kh is not None
        self.kh = kh
        self.ky = ky

    def forward(self, x, ky=None):
        if ky is None:
            ky = self.ky
        x = torch.atleast_2d(x)
        y0 = x @ self.w_xy
        h = KWTAFunction.apply(x @ self.w_xh, self.kh)
        y = KWTAFunction.apply(y0 - h @ self.w_hy, ky)
        return h, y


class IterativeWTA(nn.Module):
    def __init__(self, w_xy, w_xh, w_hy, w_yy=None, w_hh=None, w_yh=None):
        super().__init__()
        self.w_xy = w_xy
        self.w_xh = w_xh
        self.w_hy = w_hy
        self.w_yy = w_yy
        self.w_hh = w_hh
        self.w_yh = w_yh

    def forward(self, x):
        x = torch.atleast_2d(x)
        y0 = x @ self.w_xy
        h0 = x @ self.w_xh
        h = torch.zeros_like(h0, dtype=torch.int32)
        y = torch.zeros_like(y0, dtype=torch.int32)
        t_start = max(h0.max(), y0.max())
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
    def random_choice(vec):
        idx = torch.randperm(vec.shape[0])[:n_choose]
        return vec[idx]

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
    n_choose = min(n_choose, len(x_pre_idx), len(x_post_idx))
    idx_pre = random_choice(x_pre_idx)
    idx_post = random_choice(x_post_idx)
    w[idx_pre, idx_post] = 1
