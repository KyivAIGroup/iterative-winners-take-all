import torch
import torch.utils.data


def sample_bernoulli(shape, p):
    x = torch.distributions.Bernoulli(p).sample(shape)
    return x.type(torch.int32)


def random_choice(vec: torch.Tensor, n_choose: int):
    idx = torch.randperm(vec.size(0))[:n_choose]
    return vec[idx]


class NoShuffleLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, shuffle=False, **kwargs)
