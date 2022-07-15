from discopy.biclosed import Ty
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
        if len(size) > 1:
            return Ty('$%s^{%s}$' % (_label_dtype(dtype), str(size)))
        size = size[0]
    return Ty('$%s^{%d}$' % (_label_dtype(dtype), size))
