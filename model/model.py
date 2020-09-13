import collections
from discopyro import cartesian_cat, closed
import itertools
import math
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

class VaeCategoryModel(CategoryModel):
    def __init__(self, data_dim=28*28, hidden_dim=8, guide_hidden_dim=256):
        self._data_dim = data_dim

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = list(util.powers_of(2, hidden_dim, data_dim // 4)) + [data_dim]
        dims.sort()

        generators = []
        for dim_a, dim_b in itertools.combinations(dims, 2):
            lower, higher = sorted([dim_a, dim_b])
            # Construct the decoder and encoder
            if higher == self._data_dim:
                decoder = DensityDecoder(lower, higher,
                                         ContinuousBernoulliModel,
                                         convolve=True)
                encoder = DensityEncoder(higher, lower, DiagonalGaussian,
                                         convolve=True)
            else:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian)
                encoder = DensityEncoder(higher, lower, DiagonalGaussian)
            in_space, out_space = decoder.type.arrow()
            generator = closed.TypedDaggerBox(decoder.density_name, in_space,
                                              out_space, decoder, encoder,
                                              encoder.density_name)
            generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim)

class VlaeCategoryModel(CategoryModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=256):
        self._data_dim = data_dim

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = list(util.powers_of(2, hidden_dim, data_dim // 4)) + [data_dim]
        dims.sort()

        generators = []
        for lower, higher in zip(dims, dims[1:]):
            # Construct the VLAE decoder and encoder
            if higher == self._data_dim:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=True,
                                        out_dist=ContinuousBernoulliModel)
                encoder = LadderEncoder(higher, lower, DiagonalGaussian,
                                        DiagonalGaussian, noise_dim=2,
                                        conv=True)
            else:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=False,
                                        out_dist=DiagonalGaussian)
                encoder = LadderEncoder(higher, lower, DiagonalGaussian,
                                        DiagonalGaussian, noise_dim=2,
                                        conv=False)
            in_space, out_space = decoder.type.arrow()
            generator = closed.TypedDaggerBox(decoder.name, in_space, out_space,
                                              decoder, encoder, encoder.name)
            generators.append(generator)

        # For each dimensionality, construct a prior/posterior ladder pair
        for dim in set(dims) - {data_dim}:
            noise_space = types.tensor_type(torch.float, 2)
            space = types.tensor_type(torch.float, dim)
            prior = LadderPrior(2, dim, DiagonalGaussian)
            posterior = LadderPosterior(dim, 2, DiagonalGaussian)
            generator = closed.TypedDaggerBox(prior.name, noise_space, space,
                                              prior, posterior, posterior.name)
            generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim)

class GlimpseCategoryModel(CategoryModel):
    def __init__(self, data_dim=28*28, hidden_dim=4, guide_hidden_dim=256):
        self._data_dim = data_dim
        data_side = int(math.sqrt(self._data_dim))
        data_space = types.tensor_type(torch.float, data_dim)
        glimpse_side = data_side // 2
        glimpse_dim = glimpse_side ** 2
        glimpse_space = types.tensor_type(torch.float, glimpse_dim)

        latent_bernoulli = lambda dim: ContinuousBernoulliModel(dim,
                                                                'Z^{%d}' % dim)

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = list(util.powers_of(2, hidden_dim, glimpse_dim))[:-1]
        dims.sort()

        generators = []
        for dim in dims:
            in_space = types.tensor_type(torch.float, dim)
            prior = DensityDecoder(dim, glimpse_dim, latent_bernoulli,
                                   convolve=True)
            posterior = DensityEncoder(glimpse_dim, dim, DiagonalGaussian,
                                       convolve=True)
            generator = closed.TypedDaggerBox(prior.density_name, in_space,
                                              glimpse_space, prior, posterior,
                                              posterior.density_name)
            generators.append(generator)

            prior = DensityDecoder(dim, data_dim, ContinuousBernoulliModel,
                                   convolve=True)
            posterior = DensityEncoder(data_dim, dim, DiagonalGaussian,
                                       convolve=True)
            generator = closed.TypedDaggerBox(prior.density_name, in_space,
                                              data_space, prior, posterior,
                                              posterior.density_name)
            generators.append(generator)

        background = NullPrior(self._data_dim, 'X^{%d}' % self._data_dim)
        top, space = background.type.arrow()
        name = '$p(%s)$' % background.random_var_name
        global_elements = [closed.TypedBox(name, top, space, background)]

        gaze = GlimpsePrior()
        top, space = gaze.type.arrow()
        name = '$p(%s)$' % gaze.random_var_name
        global_elements.append(closed.TypedBox(name, top, space, gaze))

        # Construct writer/reader pair for spatial attention
        writer = SpatialTransformerWriter(ContinuousBernoulliModel, data_side,
                                          glimpse_side)
        writer_l, writer_r = writer.type.arrow()
        reader = SpatialTransformerReader(DiagonalGaussian,
                                          ContinuousBernoulliModel, data_side,
                                          glimpse_side)
        generator = closed.TypedDaggerBox(writer.name, writer_l, writer_r,
                                          writer, reader, reader.name)
        generators.append(generator)

        super().__init__(generators, global_elements, data_dim,
                         guide_hidden_dim)
