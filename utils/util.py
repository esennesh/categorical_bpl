import json
import math
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pyro.infer import Importance
from itertools import chain, repeat
from collections import OrderedDict
import torch
import warnings

def desymbolize(token):
    if isinstance(token, str):
        return token
    return token.symbol()

class ImportanceSampler(Importance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_likelihoods = None

    def sample(self, *args, **kwargs):
        self.log_likelihoods = []
        self.log_weights = []

        for trace, lw in self._traces(*args, **kwargs):
            self.log_likelihoods.append(trace.log_prob_sum(lambda name, site: site['is_observed']))
            self.log_weights.append(lw.detach())

    def get_log_likelihood(self):
        if self.log_likelihoods:
            log_l = torch.tensor(self.log_likelihoods)
            log_num_samples = torch.log(torch.tensor(self.num_samples * 1.0))
            return torch.logsumexp(log_l - log_num_samples, 0)
        else:
            warnings.warn(
                "The log_likelihoods list is empty, can not compute log likelihood estimate."
            )

def double_latent(ty, data_space):
    return ty if ty == data_space else ty @ ty

def double_latents(dims, data_dim):
    latents = []
    for dim in dims:
        latents.append((dim,) if dim == data_dim else (dim, dim))
    return list(chain(*latents))

def show_tensor(imgs, i=0, channels=1, shape=None):
    if shape is not None:
        imgs = imgs.view(shape)
        side = shape[-1]
    else:
        side = int(math.sqrt(imgs.shape[-1]))
    if channels in [3, 4]:
        imgs = imgs[0].view(side, side, channels)
    else:
        img = imgs[0].view(side, side)
    plt.imshow(img.cpu().detach().numpy())
    plt.show()

def powers_of(base, lower, upper):
    lower_bound = math.ceil(math.log(lower) / math.log(base))
    upper_bound = math.floor(math.log(upper) / math.log(base))
    for i in range(lower_bound, upper_bound + 1):
        yield base ** i
    yield upper

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
