import torch
import torch.utils.data


def sample_bernoulli(p, shape):
    x = torch.distributions.Bernoulli(p).sample(shape)
    return x.type(torch.int32)


class NoShuffleLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, shuffle=False, **kwargs)
