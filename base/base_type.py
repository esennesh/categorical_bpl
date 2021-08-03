from adt import adt, Case
from discopy.biclosed import Ty
import functools
import re
import torch
import uuid

def _label_dtype(dtype):
    if dtype == torch.int:
        return '\\mathbb{Z}'
    if dtype == torch.uint8:
        return 'Char'
    if dtype == torch.float:
        return '\\mathbb{R}'
    raise NotImplementedError()

def type_size(label):
    match = re.search(r'(\d+)', label)
    assert match is not None
    return int(match[0])

def tensor_type(dtype, size):
    if isinstance(size, tuple):
        return functools.reduce(lambda x, y: x @ y,
                                (tensor_type(dtype, s) for s in size))
    return Ty('$%s^{%d}$' % (_label_dtype(dtype), size))
