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


def compute_loss(tensor: torch.Tensor, labels: torch.Tensor):
    tensor = tensor / tensor.norm(dim=1, keepdim=True)
    cos = tensor.matmul(tensor.t())
    loss = []
    for label in labels.unique():
        mask_same = labels == label
        cos_same = cos[mask_same][:, mask_same]
        if cos_same.nelement() == 1:
            continue
        cos_other = cos[mask_same][:, ~mask_same]
        n = cos_same.size(0)
        ii, jj = torch.triu_indices(row=n, col=n, offset=1)
        cos_same = cos_same[ii, jj].mean()
        if cos_same != 0:
            # all vectors degenerated in a single vector
            loss.append(1 - cos_same + cos_other.mean())
    if len(loss) == 0:
        return None
    loss = torch.stack(loss).mean()
    return loss
