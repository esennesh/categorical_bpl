from discopy import Ty
from discopyro import closed
from adt import adt, Case
import torch
import uuid

def _label_dtype(dtype):
    if dtype == torch.int:
        return 'Z'
    if dtype == torch.uint8:
        return 'Char'
    if dtype == torch.float:
        return 'R'
    raise NotImplementedError()

def tensor_type(dtype, size):
    return closed.CartesianClosed.BASE(Ty('%s^{%s}' % (_label_dtype(dtype),
                                                       str(size))))
