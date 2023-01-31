# Copyright (c) 2023 Eli Sennesh
# SPDX-License-Identifier: Apache-2.0

"""
``asvi`` contains the implementation of :func:`asvi`, an implementation of
Automatic Structured Variational Inference.
"""
import collections
import math

import pyro
from pyro.distributions import Independent
from pyro.nn import PyroModule, PyroParam
from pyro.poutine.guide import GuideMessenger
from torch.distributions import transform_to
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch

from model.modules import ConvIncoder

class AsviGuide(GuideMessenger):
    """
    ``AsviGuide`` implements Automatic Structured Variational Inference as an
    effect-based guide.
    """
    def __init__(self, model, parameter_dicts, length=1, index=slice(0,-1,1)):
        super().__init__(model)
        self._mean_fields = parameter_dicts['mean_fields']
        self._prior_logits = parameter_dicts['prior_logits']
        self._length = length
        self._indices = index

    def __contains__(self, name):
        return name in self._mean_fields and name in self._prior_logits

    def initialize_parameter(self, name, **kwargs):
        self._mean_fields[name] = PyroModule[nn.ParameterDict]()
        for k, v in kwargs.items():
            self._mean_fields[name][k] = nn.Parameter(torch.zeros(
                (self._length, *v.shape[1:]), device=v.device
            ))

        self._prior_logits[name] = nn.Parameter(
            torch.zeros(self._length, device=v.device)
        )

    def get_alphas(self, name):
        return self._prior_logits[name][self._indices]

    def get_lambdas(self, name, key):
        return self._mean_fields[name][key][self._indices]

    def get_posterior(self, name, prior):
        event_shape = prior.event_shape
        if isinstance(prior, Independent):
            independent_dims = prior.reinterpreted_batch_ndims
            prior = prior.base_dist
        else:
            independent_dims = 0
        parameters = {k: v for k, v in prior.__dict__.items() if k[0] != '_'}
        if name not in self:
            self.initialize_parameter(name, **parameters)

        alphas = torch.sigmoid(self.get_alphas(name))
        alphas = alphas.reshape(alphas.shape + (1,) * len(event_shape))
        for k, v in parameters.items():
            transform = transform_to(prior.arg_constraints[k])
            lam = transform(self.get_lambdas(name, k))

            parameters[k] = alphas * v + (1 - alphas) * lam

        proposal = prior.__class__(**parameters)
        if independent_dims:
            proposal = Independent(proposal, independent_dims)
        return proposal

def asvi(model, parameter_dicts=None, **kwargs):
    return AsviGuide(model, parameter_dicts, **kwargs)

class AmortizedAsviGuide(AsviGuide):
    def __init__(self, model, net_dicts, data):
        super().__init__(model, net_dicts, length=data.shape[0])
        self._data = data
        self._dim = math.prod(self._data.shape[1:])

    def initialize_parameter(self, name, **kwargs):
        self._mean_fields[name] = PyroModule[nn.ParameterDict]()
        for k, v in kwargs.items():
            incoder = ConvIncoder(self._dim, math.prod(v.shape[1:]))
            self._mean_fields[name][k] = incoder.to(device=self._data.device)

        incoder = ConvIncoder(self._dim, 1)
        self._prior_logits[name] = incoder.to(device=self._data.device)

    def get_alphas(self, name):
        return self._prior_logits[name](self._data).squeeze()

    def get_lambdas(self, name, key):
        return self._mean_fields[name][key](self._data)

def amortized_asvi(model, module_dicts=None, data=None, **kwargs):
    return AmortizedAsviGuide(model, module_dicts, data, **kwargs)
