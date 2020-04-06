import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch

def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)

#%%
def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A,A)
    A4 = torch.matmul(A2,A2)
    A6 = torch.matmul(A4,A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

def expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))

    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings
    n_squarings = n_squarings.flatten().type(torch.int32)

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q) # solve P = Q*R

    # Unsquaring step
    expmA = [ ]
    for i in range(n_A):
        l = [R[i]]
        for _ in range(n_squarings[i]):
            l.append(l[-1].mm(l[-1]))
        expmA.append(l[-1])

    return torch.stack(expmA)

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
