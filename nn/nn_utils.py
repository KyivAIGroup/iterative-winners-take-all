import math
import torch
import torch.utils.data


def sample_bernoulli(shape, p):
    x = torch.distributions.Bernoulli(p).sample(shape)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def random_choice(vec: torch.Tensor, n_choose: int):
    idx = torch.randint(0, vec.size(0), size=(n_choose,), device=vec.device)
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


def compute_discriminative_factor(array: torch.Tensor, labels: torch.Tensor):
    # array shape is (n_classes, n_features)
    dist_nonzero = torch.pdist(array)
    n = array.size(0)
    dist = torch.zeros(n, n, device=array.device)
    ii, jj = torch.triu_indices(row=n, col=n, offset=1)
    dist[ii, jj] = dist_nonzero
    ii, jj = torch.tril_indices(row=n, col=n, offset=-1)
    dist[ii, jj] = dist_nonzero
    factor = []
    for label in labels.unique():
        mask_same = labels == label
        dist_same = dist[mask_same][:, mask_same]
        if dist_same.nelement() == 1:
            continue
        dist_other = dist[mask_same][:, ~mask_same]
        n = dist_same.size(0)
        ii, jj = torch.triu_indices(row=n, col=n, offset=1)
        dist_same = dist_same[ii, jj].mean()
        if dist_same != 0:
            # all vectors degenerated in a single vector
            factor.append(dist_other.mean() / dist_same)
    if len(factor) == 0:
        return None
    factor = torch.stack(factor).mean().item()
    return factor
