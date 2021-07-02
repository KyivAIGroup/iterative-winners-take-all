import math
import numpy as np
import torch
import torch.utils.data


def sample_bernoulli(shape, p):
    x = torch.distributions.Bernoulli(p).sample(shape)
    return x.type(torch.int32)


def random_choice(vec: torch.Tensor, n_choose: int):
    idx = np.random.choice(vec.size(0), n_choose)
    idx = torch.from_numpy(idx)
    return vec[idx]


class NoShuffleLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, shuffle=False, **kwargs)


def get_optimizer_scheduler(model):
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3,
        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=15,
                                                           threshold=1e-3,
                                                           min_lr=1e-4)
    return optimizer, scheduler


def get_kwta_threshold(tensor: torch.Tensor, sparsity: float):
    embedding_dim = tensor.shape[1]
    k_active = math.ceil(sparsity * embedding_dim)
    topk = tensor.topk(k_active + 1, dim=1).values
    threshold = topk[:, [-2, -1]].mean(dim=1)
    threshold = threshold.unsqueeze(1)
    return threshold


def l0_sparsity(tensor):
    return tensor.count_nonzero().item() / tensor.nelement()
