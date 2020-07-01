from discopy import Ty
from discopyro import cartesian_cat, closed
from indexed import IndexedOrderedDict
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pyro.nn as pnn
import pytorch_expm.expm_taylor as expm
import torch
import torch.distributions
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, TypedModel
import base.base_type as types
import utils.util as util

class ContinuousBernoulli(torch.distributions.ContinuousBernoulli,
                          dist.torch_distribution.TorchDistributionMixin):
    pass

class DiagonalGaussian(TypedModel):
    def __init__(self, dim, latent_name=None):
        super().__init__()
        self._dim = torch.Size([dim])
        if not latent_name:
            latent_name = 'Z^{%d}' % self._dim[0]
        self._latent_name = latent_name
        self.parameterization = nn.Linear(self._dim[0], self._dim[0] * 2)

    @property
    def latent_name(self):
        return self._latent_name

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._dim),
            types.tensor_type(torch.float, self._dim),
        )

    def forward(self, inputs, sample=True):
        zs = self.parameterization(inputs).view(-1, 2, self._dim[0])
        mean, std_dev = zs[:, 0], F.softplus(zs[:, 1])
        if sample:
            normal = dist.Normal(mean, std_dev).to_event(1)
            return pyro.sample(self._latent_name, normal)
        return mean, std_dev

class StandardNormal(TypedModel):
    def __init__(self, dim, latent_name=None):
        super().__init__()
        if not latent_name:
            latent_name = 'Z^{%d}' % dim
        self._latent_name = latent_name
        self._dim = dim

    @property
    def latent_name(self):
        return self._latent_name

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            closed.CartesianClosed.BASE(Ty()),
            types.tensor_type(torch.float, torch.Size([self._dim])),
        )

    def forward(self):
        z_loc = self._batch.new_zeros(torch.Size((self._batch.shape[0],
                                                  self._dim)))
        z_scale = self._batch.new_ones(torch.Size((self._batch.shape[0],
                                                   self._dim)))
        normal = dist.Normal(z_loc, z_scale).to_event(1)
        return pyro.sample(self._latent_name, normal)

class ContinuousBernoulliModel(TypedModel):
    def __init__(self, obs_dim, observable_name=None):
        super().__init__()
        self._obs_dim = torch.Size([obs_dim])
        if not observable_name:
            observable_name = 'X^{%d}' % self._obs_dim[0]
        self._observable_name = observable_name

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._obs_dim),
            types.tensor_type(torch.float, self._obs_dim),
        )

    def forward(self, inputs, sample=True):
        with name_count():
            xs = torch.sigmoid(inputs.view(-1, self._obs_dim[0]))
            if sample:
                bernoulli = ContinuousBernoulli(probs=xs).to_event(1)
                pyro.sample(self._observable_name, bernoulli)
            return xs

class PathDensityNet(TypedModel):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel):
        super().__init__()
        self._in_space = types.tensor_type(torch.float, torch.Size([in_dim]))
        self._out_space = types.tensor_type(torch.float, torch.Size([out_dim]))

        hidden_dim = in_dim + out_dim // 2
        self.add_module('residual_layer', nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.PReLU(), nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        ))
        self.add_module('projection_layer', nn.Linear(in_dim, out_dim))
        self.add_module('distribution', dist_layer(out_dim))

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(self._in_space, self._out_space)

    def forward(self, inputs, observations=None, sample=True):
        hidden = self.residual_layer(inputs) + self.projection_layer(inputs)
        return self.distribution(hidden, observations, sample)

class LayersGraph:
    def __init__(self, spaces, data_dim):
        self._prototype = nx.complete_graph(spaces)
        self._data_space = data_dim

    @property
    def spaces(self):
        return set(self._prototype.nodes())

    @property
    def latent_spaces(self):
        return {n for n in self._prototype.nodes() if n != self._data_space}

    def likelihoods(self):
        # Use the data space to construct likelihood layers
        for source in self.latent_spaces:
            yield PathDensityNet(source, self._data_space,
                                 dist_layer=ContinuousBernoulliModel)

    def encoders(self):
        # Use the data space to construct likelihood layers
        for dest in self.latent_spaces:
            yield PathDensityNet(self._data_space, dest,
                                 dist_layer=DiagonalGaussian)

    def priors(self):
        for obj in self._prototype:
            yield self.prior(obj)

    def prior(self, obj):
        return StandardNormal(obj)

    def latent_maps(self):
        for z1, z2 in itertools.permutations(self.latent_spaces, 2):
            yield PathDensityNet(z1, z2, dist_layer=DiagonalGaussian)

class VAECategoryModel(BaseModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=None):
        super().__init__()
        self._data_dim = data_dim
        if not guide_hidden_dim:
            guide_hidden_dim = data_dim // 16

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        layers_graph = LayersGraph(util.powers_of(2, hidden_dim, data_dim),
                                   data_dim)
        global_elements = [closed.TypedFunction(*prior.type.arrow(), prior)
                           for prior in layers_graph.priors()]
        likelihoods = [closed.TypedFunction(*gen.type.arrow(), gen)
                       for gen in layers_graph.likelihoods()]
        latent_maps = [closed.TypedFunction(*gen.type.arrow(), gen)
                       for gen in layers_graph.latent_maps()]
        encoders = [closed.TypedFunction(*gen.type.arrow(), gen)
                    for gen in layers_graph.encoders()]
        self._category = cartesian_cat.CartesianCategory(likelihoods +\
                                                         latent_maps + encoders,
                                                         global_elements)

        self.guide_embedding = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim),
            nn.BatchNorm1d(guide_hidden_dim), nn.PReLU(),
        )
        self.guide_confidences = nn.Sequential(
            nn.Linear(guide_hidden_dim, 1 * 2), nn.Softplus(),
        )

        self.guide_navigator = nn.GRUCell(len(self._category.obs),
                                          guide_hidden_dim)
        self.guide_navigation_decoder = nn.Sequential(
            nn.Linear(guide_hidden_dim, len(self._category.ars)), nn.Softplus()
        )

    @property
    def data_space(self):
        return types.tensor_type(torch.float, torch.Size([self._data_dim]))

    @pnn.pyro_method
    def model(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        if data is None:
            data = torch.zeros(1, self._data_dim)
        data = data.view(data.shape[0], self._data_dim)
        for module in self._category.children():
            module.set_batching(data)

        morphism = pyro.condition(self._category(self.data_space, min_depth=1),
                                  data={'X^{%d}' % self._data_dim: data})
        with pyro.plate('data', len(data)):
            with name_count():
                output = morphism()
        return morphism, output

    @pnn.pyro_method
    def guide(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], self._data_dim)
        for module in self._category.children():
            module.set_batching(data)

        embedding = self.guide_embedding(data).mean(dim=0)

        confidences = self.guide_confidences(embedding).view(1, 2)

        confidence_gamma = dist.Gamma(confidences[0, 0],
                                      confidences[0, 1]).to_event(0)
        confidence = pyro.sample('distances_confidence', confidence_gamma)

        morphism = self._category(self.data_space, min_depth=1,
                                  confidence=confidence)
        with pyro.plate('data', len(data)):
            with name_count():
                morphism()

        return morphism

    def forward(self, observations=None):
        if observations is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(
                observations=observations
            )
            return pyro.poutine.replay(self.model, trace=trace)(
                observations=observations
            )
        else:
            return self.model(observations=None)
