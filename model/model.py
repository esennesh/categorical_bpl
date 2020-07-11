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
    def random_var_name(self):
        return self._latent_name

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._dim),
            types.tensor_type(torch.float, self._dim),
        )

    def forward(self, inputs):
        zs = self.parameterization(inputs).view(-1, 2, self._dim[0])
        mean, std_dev = zs[:, 0], F.softplus(zs[:, 1])
        normal = dist.Normal(mean, std_dev).to_event(1)
        return pyro.sample(self._latent_name, normal)

class StandardNormal(TypedModel):
    def __init__(self, dim, latent_name=None):
        super().__init__()
        if not latent_name:
            latent_name = 'Z^{%d}' % dim
        self._latent_name = latent_name
        self._dim = dim

    @property
    def random_var_name(self):
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
    def random_var_name(self):
        return self._observable_name

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._obs_dim),
            types.tensor_type(torch.float, self._obs_dim),
        )

    def forward(self, inputs):
        with name_count():
            xs = torch.sigmoid(inputs.view(-1, self._obs_dim[0]))
            bernoulli = ContinuousBernoulli(probs=xs).to_event(1)
            pyro.sample(self._observable_name, bernoulli)
            return xs

class DensityNet(TypedModel):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel,
                 normalizer_layer=nn.LayerNorm):
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._in_space = types.tensor_type(torch.float, torch.Size([in_dim]))
        self._out_space = types.tensor_type(torch.float, torch.Size([out_dim]))

        hidden_dim = (in_dim + out_dim) // 2
        self.add_module('neural_layers', nn.Sequential(
            nn.Linear(in_dim, hidden_dim), normalizer_layer(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), normalizer_layer(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim), normalizer_layer(out_dim),
        ))
        self.add_module('distribution', dist_layer(out_dim))

    def set_batching(self, batch):
        super().set_batching(batch)
        self.distribution.set_batching(batch)

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(self._in_space, self._out_space)

    @property
    def density_name(self):
        sample_name = self.distribution.random_var_name
        condition_name = 'Z^{%d}' % self._in_dim
        return 'p(%s | %s)' % (sample_name, condition_name)

class DensityDecoder(DensityNet):
    def __init__(self, in_dim, out_dim, latent=True,
                 dist_layer=ContinuousBernoulliModel):
        super().__init__(in_dim, out_dim, dist_layer)
        self._latent = latent
        if self._latent:
            self.add_module('combination_layer', nn.Sequential(
                nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim),
                nn.PReLU(),
                nn.Linear(out_dim, out_dim)
            ))

    def forward(self, inputs):
        hidden = self.neural_layers(inputs)
        if self._latent:
            noise = self.distribution()
            return self.combination_layer(torch.cat((hidden, noise), dim=-1))
        return self.distribution(hidden)

class DensityEncoder(DensityNet):
    def __init__(self, in_dim, out_dim, dist_layer=DiagonalGaussian):
        super().__init__(in_dim, out_dim, dist_layer)

    @property
    def density_name(self):
        sample_name = self.distribution.random_var_name
        condition_name = 'Z^{%d}' % self._in_dim
        return 'q(%s | %s)' % (sample_name, condition_name)

    def forward(self, inputs):
        out_hidden = self.neural_layers(inputs)
        self.distribution(out_hidden)
        return out_hidden

class VAECategoryModel(BaseModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=None):
        super().__init__()
        self._data_dim = data_dim
        if not guide_hidden_dim:
            guide_hidden_dim = data_dim // 16

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = list(util.powers_of(2, hidden_dim, data_dim))

        generators = []
        for dim_a, dim_b in itertools.combinations(dims, 2):
            lower, higher = sorted([dim_a, dim_b])
            # Construct the decoder
            if higher == self._data_dim:
                decoder = DensityDecoder(lower, higher, False,
                                         ContinuousBernoulliModel)
            else:
                decoder = DensityDecoder(lower, higher, True,
                                         StandardNormal)
            # Construct the encoder
            encoder = DensityEncoder(higher, lower, DiagonalGaussian)
            in_space, out_space = decoder.type.arrow()
            generator = closed.TypedDaggerBox(decoder.density_name, in_space,
                                              out_space, decoder, encoder)
            generators.append(generator)

        global_elements = []
        for dim in dims:
            space = types.tensor_type(torch.float, torch.Size([dim]))
            prior = StandardNormal(dim)
            name = 'p(%s)' % prior.random_var_name
            global_element = closed.TypedBox(name, closed.TOP, space, prior)
            global_elements.append(global_element)

        self._category = cartesian_cat.CartesianCategory(generators,
                                                         global_elements)

        self.guide_embedding = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim, guide_hidden_dim), nn.PReLU(),
        )
        self.guide_confidences = nn.Sequential(
            nn.Linear(guide_hidden_dim, 1 * 2), nn.Softplus(),
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

        morphism = self._category(self.data_space, min_depth=2)
        if observations is not None:
            conditions = {'X^{%d}' % self._data_dim: data}
            score_morphism = pyro.condition(morphism, data=conditions)
        else:
            score_morphism = morphism
        with pyro.plate('data', len(data)):
            with name_count():
                output = score_morphism()
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
