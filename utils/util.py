import json
import math
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch

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
