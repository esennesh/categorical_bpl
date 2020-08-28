# Copyright (c) 2020 Eli Sennesh
# SPDX-License-Identifier: Apache-2.0

"""
``name_stack`` contains the implementation of :func:`name_stack`, a tool for
automatically numbering sample sites according to a stack.
"""
import collections

from pyro.poutine.messenger import Messenger

def _count_stacked(stack, name, offset=0):
    count = max(stack[name] + offset, 0)
    if count:
        return name + "__%d" % count
    return name

def _stacked_name_base(name):
    split_name = name.split('__')
    if '__' in name and split_name[-1].isdigit():
        return '__'.join(split_name[:-1])
    return name

class NamePushMessenger(Messenger):
    """
    ``NamePushMessenger`` is the implementation of :func:`name_push`
    """
    def __init__(self, name_stack=None):
        super().__init__()
        if name_stack is not None:
            self._names = name_stack
        else:
            name_stack = collections.defaultdict(int)

    def _pyro_sample(self, msg):
        offset = int(not msg["is_observed"])
        msg["name"] = _count_stacked(self._names, msg["name"], offset=offset)

    def _pyro_post_sample(self, msg):
        base_name = _stacked_name_base(msg["name"])
        self._names[base_name] += 1

class NamePopMessenger(Messenger):
    """
    ``NamePopMessenger`` is the implementation of :func:`name_pop`
    """
    def __init__(self, name_stack=None):
        super().__init__()
        if name_stack is not None:
            self._names = name_stack
        else:
            name_stack = collections.defaultdict(int)

    def _pyro_sample(self, msg):
        name = msg["name"]
        msg["name"] = _count_stacked(self._names, name)
        if self._names[name]:
            msg["value"] = None
            msg["is_observed"] = False

    def _pyro_post_sample(self, msg):
        base_name = _stacked_name_base(msg["name"])
        if self._names[base_name]:
            self._names[base_name] -= 1

def name_push(fn=None, name_stack=None):
    msngr = NamePushMessenger(name_stack)
    return msngr(fn) if fn is not None else msngr

def name_pop(fn=None, name_stack=None):
    msngr = NamePopMessenger(name_stack)
    return msngr(fn) if fn is not None else msngr
