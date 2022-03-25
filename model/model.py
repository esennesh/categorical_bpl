import collections
from discopy.biclosed import Ty
from discopy import cat, wiring
from discopyro import cart_closed, freecat, unification
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

WIRING_FUNCTOR = wiring.WiringFunctor(True)

def latent_effect_falgebra(f):
    if isinstance(f, wiring.Id):
        return []
    if isinstance(f, wiring.Box):
        return [eff for eff in f.data['effect'] if 'X^' not in eff]
    if isinstance(f, wiring.Parallel):
        return list(itertools.chain(*f.factors))
    if isinstance(f, wiring.Sequential):
        return list(itertools.chain(*f.arrows))
    raise TypeError('Expected wiring diagram', f)

class CategoryModel(BaseModel):
    def __init__(self, generators, global_elements=[], data_space=(784,),
                 guide_hidden_dim=256, no_prior_dims=[]):
        super().__init__()
        if isinstance(data_space, int):
            data_space = (data_space,)
        self._data_space = data_space
        self._data_dim = math.prod(data_space)
        if len(self._data_space) == 1:
            self._observation_name = '$X^{%d}$' % self._data_dim
        else:
            self._observation_name = '$X^{%s}$' % str(self._data_space)

        obs = set()
        for generator in generators:
            ty = generator.dom >> generator.cod
            obs = obs | unification.base_elements(ty)
        for element in global_elements:
            ty = element.dom >> element.cod
            obs = obs - unification.base_elements(ty)

        no_prior_dims = no_prior_dims + [self._data_dim]
        for ob in obs:
            dim = types.type_size(str(ob))
            if dim in no_prior_dims:
                continue

            space = types.tensor_type(torch.float, dim)
            prior = StandardNormal(dim)
            name = '$p(%s)$' % prior.effects
            effect = {'effect': prior.effect, 'dagger_effect': []}
            global_element = cart_closed.Box(name, Ty(), space, prior,
                                             data=effect)
            global_elements.append(global_element)

        self._category = freecat.FreeCategory(generators, global_elements)

        self.guide_temperatures = nn.Sequential(
            nn.Linear(self._data_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim, 1 * 2), nn.Softplus(),
        )
        self.guide_arrow_weights = nn.Sequential(
            nn.Linear(self._data_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim,
                      self._category.arrow_weight_alphas.shape[0] * 2),
            nn.Softplus()
        )

        self._random_variable_names = collections.defaultdict(int)

        self.encoders = nn.ModuleDict()
        self.encoder_functor = wiring.Functor(
            lambda ty: util.double_latent(ty, self.data_space),
            lambda ar: self._encoder(ar.name), ob_factory=Ty,
            ar_factory=cart_closed.Box
        )

        for arrow in self._category.ars:
            effect = [eff for eff in arrow.data['effect'] if 'X^' not in eff]

            cod_dims = util.double_latents([types.type_size(ob.name) for ob in
                                            arrow.cod], self._data_dim)
            dom_dims = util.double_latents([types.type_size(ob.name) for ob in
                                            arrow.dom], self._data_dim)
            self.encoders[arrow.name + '†'] = build_encoder(cod_dims, dom_dims,
                                                            effect)

    def _encoder(self, name):
        encoder = self.encoders[name]
        return cart_closed.Box(
            name, encoder.type.left, encoder.type.right, encoder,
            data={'effect': encoder.effect}
        )

    @property
    def data_space(self):
        return types.tensor_type(torch.float, self._data_space)

    @pnn.pyro_method
    def model(self, observations=None, train=True):
        if isinstance(observations, dict):
            data = observations[self._observation_name]
        elif observations is not None:
            data = observations
            observations = {
                self._observation_name: observations.view(-1, *self._data_space)
            }
        else:
            data = torch.zeros(1, *self._data_space)
            observations = {}
        data = data.view(data.shape[0], *self._data_space)
        for module in self._category.children():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        morphism = self._category(wiring.Box('', Ty(), self.data_space,
                                             data={'effect': lambda e: True}),
                                  min_depth=VAE_MIN_DEPTH)
        if observations is not None and train:
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
        data = data.view(data.shape[0], *self._data_space)
        flat_data = data.view(data.shape[0], self._data_dim)
        for module in self._category.children():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        temperatures = self.guide_temperatures(flat_data).mean(dim=0).view(1, 2)
        temperature_gamma = dist.Gamma(temperatures[0, 0],
                                       temperatures[0, 1]).to_event(0)
        temperature = pyro.sample('weights_temperature', temperature_gamma)

        data_arrow_weights = self.guide_arrow_weights(flat_data)
        data_arrow_weights = data_arrow_weights.mean(dim=0).view(-1, 2)
        arrow_weights = pyro.sample(
            'arrow_weights',
            dist.Gamma(data_arrow_weights[:, 0],
                       data_arrow_weights[:, 1]).to_event(1)
        )

        morphism = self._category(wiring.Box('', Ty(), self.data_space,
                                             data={'effect': lambda e: True}),
                                  min_depth=VAE_MIN_DEPTH,
                                  temperature=temperature,
                                  arrow_weights=arrow_weights)

        wires = WIRING_FUNCTOR(morphism.dagger())
        dagger = self.encoder_functor(wires)
        with pyro.plate('data', len(data)):
            with name_push(name_stack=self._random_variable_names):
                dagger(data)

        return morphism

    def forward(self, observations=None, train=True):
        if observations is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(
                observations=observations
            )
            return pyro.poutine.replay(self.model, trace=trace)(
                observations=observations, train=train
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
            else:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian)
            data = {'effect': decoder.effect}
            generator = cart_closed.Box(decoder.density_name, decoder.type.left,
                                        decoder.type.right, decoder, data=data)
            generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim)

class VlaeCategoryModel(CategoryModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=256):
        self._data_dim = data_dim

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = list(util.powers_of(2, hidden_dim, data_dim // 4)) + [data_dim]
        dims.sort()

        gaussian_likelihood = lambda dim: DiagonalGaussian(
            dim, latent_name='X^{%d}' % dim, likelihood=True
        )

        generators = []
        for lower, higher in zip(dims, dims[1:]):
            # Construct the VLAE decoder and encoder
            if higher == self._data_dim:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=True,
                                        out_dist=gaussian_likelihood)
            else:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=False,
                                        out_dist=None)
            data = {'effect': decoder.effect}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, decoder, data=data)
            generators.append(generator)

        # For each dimensionality, construct a prior/posterior ladder pair
        for dim in set(dims) - {data_dim}:
            space = types.tensor_type(torch.float, dim)
            prior = LadderPrior(dim, None)

            generator = cart_closed.Box(prior.name, Ty(), space, prior,
                                        data={'effect': prior.effect})
            generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim,
                         list(set(dims) - {data_dim}))

class GlimpseCategoryModel(CategoryModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=256):
        self._data_dim = data_dim
        data_side = int(math.sqrt(self._data_dim))
        glimpse_side = data_side // 2
        glimpse_dim = glimpse_side ** 2

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and glimpse_dim.
        dims = list(util.powers_of(2, hidden_dim, glimpse_dim // 4)) +\
               [glimpse_dim]
        dims.sort()

        generators = []
        for dim_a, dim_b in itertools.combinations(dims, 2):
            lower, higher = sorted([dim_a, dim_b])
            # Construct the decoder and encoder
            if higher == glimpse_dim:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian,
                                         convolve=True)
                encoder = DensityEncoder(higher, lower, DiagonalGaussian,
                                         convolve=True)
            else:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian)
                encoder = DensityEncoder(higher, lower, DiagonalGaussian)
            data = {'effect': decoder.effect, 'dagger_effect': encoder.effect}
            generator = cart_closed.Box(decoder.density_name, decoder.type.left,
                                        decoder.type.right, decoder, data=data)
            generators.append(generator)

            data = {'effect': encoder.effect, 'dagger_effect': decoder.effect}
            generator = cart_closed.Box(encoder.density_name, encoder.type.left,
                                        encoder.type.right, encoder, data=data)
            generators.append(generator)

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = dims + [data_dim]
        dims.sort()

        gaussian_likelihood = lambda dim: DiagonalGaussian(
            dim, latent_name='X^{%d}' % dim
        )

        generators = []
        for lower, higher in zip(dims, dims[1:]):
            # Construct the VLAE decoder and encoder
            if higher == self._data_dim:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=True,
                                        out_dist=gaussian_likelihood)
                encoder = LadderEncoder(higher, lower, DiagonalGaussian,
                                        DiagonalGaussian, noise_dim=2,
                                        conv=True)
            else:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=False,
                                        out_dist=DiagonalGaussian)
                encoder = LadderEncoder(higher, lower, DiagonalGaussian,
                                        DiagonalGaussian, noise_dim=2,
                                        conv=False)
            data = {'effect': decoder.effect, 'dagger_effect': encoder.effect}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, decoder, data=data)
            generators.append(generator)

            data = {'effect': encoder.effect, 'dagger_effect': decoder.effect}
            generator = cart_closed.Box(encoder.name, encoder.type.left,
                                        encoder.type.right, encoder, data=data)
            generators.append(generator)

        # For each dimensionality, construct a prior/posterior ladder pair
        for dim in set(dims) - {glimpse_dim, data_dim}:
            space = types.tensor_type(torch.float, dim)
            prior = LadderPrior(dim, DiagonalGaussian)
            posterior = LadderPosterior(dim, DiagonalGaussian)

            data = {'effect': prior.effect, 'dagger_effect': posterior.effect}
            generator = cart_closed.Box(prior.name, Ty(), space, prior,
                                        data=data)
            generators.append(generator)

            data = {'effect': posterior.effect, 'dagger_effect': prior.effect}
            generator = cart_closed.Box(posterior.name, space, Ty(),
                                        posterior, data=data)
            generators.append(generator)

        # Construct writer/reader pair for spatial attention
        writer = SpatialTransformerWriter(data_side, glimpse_side)
        writer_l, writer_r = writer.type.left, writer.type.right
        reader = SpatialTransformerReader(data_side, glimpse_side)

        data = {'effect': writer.effect, 'dagger_effect': reader.effect}
        generator = cart_closed.Box(writer.name, writer_l, writer_r, writer,
                                    data=data)
        generators.append(generator)

        data = {'effect': reader.effect, 'dagger_effect': writer.effect}
        generator = cart_closed.Box(reader.name, reader.type.left,
                                    reader.type.right, reader, data=data)
        generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim,
                         [glimpse_dim])

class MolecularVaeCategoryModel(CategoryModel):
    def __init__(self, max_len=120, guide_hidden_dim=256, charset_len=34):
        hidden_dims = [196, 292, 435]
        recurrent_dims = [64, 128, 256]
        generators = []
        dagger_generators = []

        for hidden in hidden_dims:
            for recurrent in recurrent_dims:
                encoder = ConvMolecularEncoder(hidden, charset_len, max_len)
                decoder = MolecularDecoder(hidden, recurrent_dim=recurrent,
                                           charset_len=charset_len,
                                           max_len=max_len)
                data = {'effect': decoder.effect,
                        'dagger_effect': encoder.effect}
                conv_generator = cart_closed.Box(decoder.name,
                                                 decoder.type.left,
                                                 decoder.type.right, decoder,
                                                 data=data)
                generators.append(conv_generator)
                data = {'dagger_effect': decoder.effect,
                        'effect': encoder.effect}
                conv_dagger = cart_closed.Box(encoder.name, encoder.type.left,
                                              encoder.type.right, encoder,
                                              data=data)
                dagger_generators.append(conv_dagger)

                encoder = RecurrentMolecularEncoder(hidden, recurrent,
                                                    charset_len, max_len)
                decoder = MolecularDecoder(hidden, recurrent_dim=recurrent,
                                           charset_len=charset_len,
                                           max_len=max_len)
                data = {'effect': decoder.effect,
                        'dagger_effect': encoder.effect}
                rec_generator = cart_closed.Box(decoder.name, decoder.type.left,
                                                decoder.type.right, decoder,
                                                data=data)
                generators.append(rec_generator)
                data = {'dagger_effect': decoder.effect,
                        'effect': encoder.effect}
                rec_dagger = cart_closed.Box(encoder.name, encoder.type.left,
                                             encoder.type.right, encoder,
                                             data=data)
                dagger_generators.append(rec_dagger)

        super().__init__(generators, [], data_space=(max_len, charset_len),
                         guide_hidden_dim=guide_hidden_dim,
                         no_prior_dims=[max_len, charset_len],
                         dagger_generators=dagger_generators)
