import collections
from discopyro import cartesian_cat, closed
import itertools
import matplotlib.pyplot as plt
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import base.base_type as types
from model.modules import *
import utils.util as util
from utils.name_stack import name_push, name_pop

VAE_MIN_DEPTH = 2

class CategoryModel(BaseModel):
    def __init__(self, generators, global_elements=[], data_dim=28*28,
                 guide_hidden_dim=256):
        super().__init__()
        self._data_dim = data_dim
        self._observation_name = '$X^{%d}$' % self._data_dim

        combination = LinearCombination(self._data_dim,
                                        ContinuousBernoulliModel)
        decombination = LinearDecombination(self._data_dim,
                                            ContinuousBernoulliModel)
        combination_l, combination_r = combination.type.arrow()
        generator = closed.TypedDaggerBox(combination.name, combination_l,
                                          combination_r, combination,
                                          decombination, decombination.name)
        generators.append(generator)

        obs = set()
        for generator in generators:
            obs = obs | generator.type.base_elements()

        for ob in obs:
            dim = types.type_size(ob.name)
            if dim == self._data_dim:
                continue

            space = types.tensor_type(torch.float, dim)
            prior = StandardNormal(dim)
            name = '$p(%s)$' % prior.random_var_name
            global_element = closed.TypedBox(name, closed.TOP, space, prior)
            global_elements.append(global_element)

        self._category = cartesian_cat.CartesianCategory(generators,
                                                         global_elements)

        self.guide_temperatures = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim, 1 * 2), nn.Softplus(),
        )
        self.guide_arrow_weights = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim,
                      self._category.arrow_weight_alphas.shape[0] * 2),
            nn.Softplus()
        )

        self._random_variable_names = collections.defaultdict(int)

    @property
    def data_space(self):
        return types.tensor_type(torch.float, self._data_dim)

    @pnn.pyro_method
    def model(self, observations=None):
        if isinstance(observations, dict):
            data = observations[self._observation_name]
        elif observations is not None:
            data = observations
            observations = {
                self._observation_name: observations.view(-1, self._data_dim)
            }
        else:
            data = torch.zeros(1, self._data_dim)
            observations = {}
        data = data.view(data.shape[0], self._data_dim)
        for module in self._category.children():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        morphism = self._category(self.data_space, min_depth=VAE_MIN_DEPTH)
        if observations is not None:
            score_morphism = pyro.condition(morphism, data=observations)
        else:
            score_morphism = morphism
        with pyro.plate('data', len(data)):
            with name_pop(name_stack=self._random_variable_names):
                output = score_morphism()
        return morphism, output

    @pnn.pyro_method
    def guide(self, observations=None):
        if isinstance(observations, dict):
            data = observations['$X^{%d}$' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], self._data_dim)
        for module in self._category.children():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        temperatures = self.guide_temperatures(data).mean(dim=0).view(1, 2)
        temperature_gamma = dist.Gamma(temperatures[0, 0],
                                       temperatures[0, 1]).to_event(0)
        temperature = pyro.sample('weights_temperature', temperature_gamma)

        data_arrow_weights = self.guide_arrow_weights(data)
        data_arrow_weights = data_arrow_weights.mean(dim=0).view(-1, 2)
        arrow_weights = pyro.sample(
            'arrow_weights',
            dist.Gamma(data_arrow_weights[:, 0],
                       data_arrow_weights[:, 1]).to_event(1)
        )

        morphism = self._category(self.data_space, min_depth=VAE_MIN_DEPTH,
                                  temperature=temperature,
                                  arrow_weights=arrow_weights)
        with pyro.plate('data', len(data)):
            with name_push(name_stack=self._random_variable_names):
                morphism[::-1](data)

        return morphism

    def forward(self, observations=None):
        if observations is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(
                observations=observations
            )
            return pyro.poutine.replay(self.model, trace=trace)(
                observations=observations
            )
        return self.model(observations=None)
