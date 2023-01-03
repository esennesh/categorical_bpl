# Copyright (c) 2023 Eli Sennesh
# SPDX-License-Identifier: Apache-2.0

"""
``asvi`` contains the implementation of :func:`asvi`, an implementation of
Automatic Structured Variational Inference.
"""
import collections

import pyro
from pyro.distributions import Independent
from pyro.nn import PyroModule, PyroParam
from pyro.poutine.guide import GuideMessenger
from torch.distributions import transform_to
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch

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

        alphas = torch.sigmoid(self._prior_logits[name][self._indices])
        alphas = alphas.reshape(alphas.shape + (1,) * len(event_shape))
        for k, v in parameters.items():
            transform = transform_to(prior.arg_constraints[k])
            lam = transform(self._mean_fields[name][k][self._indices])

            parameters[k] = alphas * v + (1 - alphas) * lam

        proposal = prior.__class__(**parameters)
        if independent_dims:
            proposal = Independent(proposal, independent_dims)
        return proposal

def asvi(model, parameter_dicts=None, **kwargs):
    return AsviGuide(model, parameter_dicts, **kwargs)
