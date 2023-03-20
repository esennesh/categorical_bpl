import collections
from discopy.python import Ty
from discopy import monoidal, wiring
from discopyro import free_operad, unification
import itertools
import math
import matplotlib.pyplot as plt
import pyro
from pyro.contrib.autoname import name_count, scope
from pyro.poutine import block
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import base.base_type as types
from model.asvi import *
from model.modules import *
import model.modules as modules
from utils.name_stack import name_push, name_pop
import utils.util as util

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

class OperadicModel(BaseModel):
    def __init__(self, generators, global_elements=[], data_space=(784,),
                 guide_in_dim=None, guide_hidden_dim=256):
        super().__init__()
        if isinstance(data_space, int):
            data_space = (data_space,)
        self._data_space = data_space
        self._data_dim = math.prod(data_space)
        if not guide_in_dim:
            guide_in_dim = self._data_dim
        if len(self._data_space) == 1:
            self._observation_name = '$X^{%d}$' % self._data_dim
        else:
            self._observation_name = '$X^{%s}$' % str(self._data_space)

        self._operad = free_operad.FreeOperad(generators, global_elements)

        self.guide_temperatures = nn.Sequential(
            nn.Linear(guide_in_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim, 1 * 2), nn.Softplus(),
        )
        self.guide_arrow_weights = nn.Sequential(
            nn.Linear(guide_in_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim,
                      self._operad.arrow_weight_loc.shape[0] * 2),
        )
        self._random_variable_names = collections.defaultdict(int)

    @property
    def data_space(self):
        return types.tensor_type(torch.float, self._data_space)

    @property
    def wiring_diagram(self):
        return wiring.Box('', Ty(), self.data_space,
                          data={'effect': lambda e: True})

    @pnn.pyro_method
    def model(self, observations=None, **kwargs):
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
        for module in self.modules():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        min_depth = VAE_MIN_DEPTH if len(list(self.wiring_diagram)) == 1 else 0
        morphism = self._operad(self.wiring_diagram, min_depth=min_depth)

        return morphism, observations, data

    @pnn.pyro_method
    def guide(self, observations=None, summary=None, **kwargs):
        if isinstance(observations, dict):
            data = observations['$X^{%d}$' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], *self._data_space)
        if summary is None:
            summary = data.view(data.shape[0], self._data_dim)
        for module in self._operad.children():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        temperatures = self.guide_temperatures(summary).mean(dim=0).view(1, 2)
        temperature_gamma = dist.Gamma(temperatures[0, 0],
                                       temperatures[0, 1]).to_event(0)
        temperature = pyro.sample('weights_temperature', temperature_gamma)

        data_arrow_weights = self.guide_arrow_weights(summary)
        data_arrow_weights = data_arrow_weights.mean(dim=0).view(-1, 2)
        arrow_weights = pyro.sample(
            'arrow_weights',
            dist.Normal(data_arrow_weights[:, 0],
                        data_arrow_weights[:, 1].exp()).to_event(1)
        )

        min_depth = VAE_MIN_DEPTH if len(list(self.wiring_diagram)) == 1 else 0
        morphism = self._operad(self.wiring_diagram, min_depth=min_depth,
                                  temperature=temperature,
                                  arrow_weights=arrow_weights)

        return morphism, data

    def forward(self, **kwargs):
        if 'observations' in kwargs and kwargs['observations'] is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(**kwargs)
            return pyro.poutine.replay(self.model, trace=trace)(**kwargs)
        return self.model(**kwargs)

class DaggerOperadicModel(OperadicModel):
    def __init__(self, generators, global_elements=[], data_space=(784,),
                 guide_hidden_dim=256, no_prior_dims=[]):
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
            data = {'effect': prior.effect, 'dagger_effect': [],
                    'function': prior}
            global_element = cart_closed.Box(name, Ty(), space, data=data)
            global_elements.append(global_element)

        super().__init__(generators, global_elements, data_space,
                         guide_hidden_dim=guide_hidden_dim)

        self.encoders = nn.ModuleDict()
        self.encoder_functor = wiring.Functor(
            lambda ty: util.double_latent(ty, self.data_space),
            lambda ar: self._encoder(ar.name)
        )

        for arrow in self._operad.ars:
            effect = arrow.data['effect']

            cod_dims = util.double_latents([types.type_size(ob.name) for ob in
                                            arrow.cod], self._data_dim)
            dom_dims = util.double_latents([types.type_size(ob.name) for ob in
                                            arrow.dom], self._data_dim)
            self.encoders[arrow.name + '†'] = build_encoder(cod_dims, dom_dims,
                                                            effect)

    def _encoder(self, name):
        encoder = self.encoders[name]
        return cart_closed.Box(
            name, encoder.type.left, encoder.type.right,
            data={'effect': encoder.effect, 'function': encoder}
        )

    @pnn.pyro_method
    def model(self, observations=None, **kwargs):
        morphism, observations, data = super().model(observations)

        if observations is not None:
            score_morphism = pyro.condition(morphism, data=observations)
        else:
            score_morphism = morphism
        with pyro.plate('data', len(data)):
            with name_pop(name_stack=self._random_variable_names):
                output = score_morphism()
        return morphism, output

    @pnn.pyro_method
    def guide(self, observations=None, **kwargs):
        morphism, data = super().guide(observations)

        wires = WIRING_FUNCTOR(morphism.dagger())
        dagger = self.encoder_functor(wires)
        with pyro.plate('data', len(data)):
            with name_push(name_stack=self._random_variable_names):
                dagger(data)

        return morphism

class VaeOperadicModel(DaggerOperadicModel):
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
            data = {'effect': decoder.effect, 'function': decoder}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, data=data)
            generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim)

class VlaeOperadicModel(DaggerOperadicModel):
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
            data = {'effect': decoder.effect, 'function': decoder}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, data=data)
            generators.append(generator)

        # For each dimensionality, construct a prior/posterior ladder pair
        for dim in set(dims) - {data_dim}:
            space = types.tensor_type(torch.float, dim)
            prior = LadderPrior(dim, None)

            generator = cart_closed.Box(prior.name, Ty(), space, data={
                'effect': prior.effect, 'function': prior
            })
            generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim,
                         list(set(dims) - {data_dim}))

class GlimpseOperadicModel(DaggerOperadicModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=256):
        self._data_dim = data_dim
        data_side = int(math.sqrt(self._data_dim))
        glimpse_side = data_side // 2
        glimpse_dim = glimpse_side ** 2

        generators = []

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and glimpse_dim.
        dims = list(util.powers_of(2, hidden_dim, glimpse_dim // 4)) +\
               [glimpse_dim]
        dims.sort()

        for dim_a, dim_b in itertools.combinations(dims, 2):
            lower, higher = sorted([dim_a, dim_b])
            # Construct the decoder and encoder
            if higher == glimpse_dim:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian,
                                         convolve=True)
            else:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian)
            data = {'effect': decoder.effect, 'function': decoder}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, data=data)
            generators.append(generator)

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = dims + [data_dim]
        dims.sort()

        for lower, higher in zip(dims, dims[1:]):
            # Construct the VLAE decoder and encoder
            if higher == self._data_dim:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=True,
                                        out_dist=DiagonalGaussian)
            else:
                decoder = LadderDecoder(lower, higher, noise_dim=2, conv=False,
                                        out_dist=DiagonalGaussian)
            data = {'effect': decoder.effect, 'function': decoder}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, data=data)
            generators.append(generator)

        # For each dimensionality, construct a prior/posterior ladder pair
        for dim in set(dims) - {glimpse_dim, data_dim}:
            space = types.tensor_type(torch.float, dim)
            prior = LadderPrior(dim, DiagonalGaussian)

            data = {'effect': prior.effect, 'function': prior}
            generator = cart_closed.Box(prior.name, Ty(), space, data=data)
            generators.append(generator)

        # Construct writer/reader pair for spatial attention
        writer = SpatialTransformerWriter(data_side, glimpse_side)
        writer_l, writer_r = writer.type.left, writer.type.right

        data = {'effect': writer.effect, 'function': writer}
        generator = cart_closed.Box(writer.name, writer_l, writer_r, data=data)
        generators.append(generator)

        # Construct the likelihood
        likelihood = GaussianLikelihood(data_dim, 'X^{%d}' % data_dim)
        data = {'effect': likelihood.effect, 'function': likelihood}
        generator = cart_closed.Box(likelihood.name, likelihood.type.left,
                                    likelihood.type.right, data=data)
        generators.append(generator)

        super().__init__(generators, [], data_dim, guide_hidden_dim,
                         no_prior_dims=[glimpse_dim, data_dim])

    @property
    def wiring_diagram(self):
        latent = super().wiring_diagram
        observation_effect = 'X^{%d}' % self._data_dim
        likelihood = wiring.Box('', self.data_space, self.data_space,
                                data={'effect': [observation_effect]})
        return latent >> likelihood

class AutoencodingOperadicModel(OperadicModel):
    def __init__(self, generators, latent_space=(64,), global_elements=[],
                 data_space=(784,), guide_hidden_dim=256):
        if isinstance(latent_space, int):
            latent_space = (latent_space,)
        self._latent_space = latent_space
        self._latent_dim = math.prod(latent_space)
        if len(self._latent_space) == 1:
            self._latent_name = 'Z^{%d}' % self._latent_dim
        else:
            self._latent_name = 'Z^{%s}' % str(self._latent_space)

        super().__init__(generators, global_elements, data_space,
                         guide_in_dim=self._latent_dim,
                         guide_hidden_dim=guide_hidden_dim)

        space = types.tensor_type(torch.float, latent_space)
        self.latent_prior = StandardNormal(math.prod(latent_space),
                                           self._latent_name)

    @property
    def latent_space(self):
        return types.tensor_type(torch.float, self._latent_space)

    @property
    def wiring_diagram(self):
        return wiring.Box('', self.latent_space, self.data_space,
                          data={'effect': lambda e: True})

    @pnn.pyro_method
    def model(self, observations=None):
        morphism, observations, data = super().model(observations)
        latent_code = self.latent_prior()

        if observations is not None:
            score_morphism = pyro.condition(morphism, data=observations)
        else:
            score_morphism = morphism
        with pyro.plate('data', len(data)):
            with name_pop(name_stack=self._random_variable_names):
                output = score_morphism(latent_code)
        return morphism, output

    @pnn.pyro_method
    def guide(self, observations=None):
        if isinstance(observations, dict):
            data = observations['$X^{%d}$' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], *self._data_space)
        latent_code = self.encoder(data)
        morphism, data = super().guide(observations, latent_code)

        with pyro.plate('data', len(data)):
            with name_push(name_stack=self._random_variable_names):
                morphism(latent_code)

        return morphism, latent_code

class AsviOperadicModel(OperadicModel):
    def __init__(self, generators, global_elements=[], data_space=(784,),
                 guide_hidden_dim=256, dataset_length=1, amortized=False):
        super().__init__(generators, global_elements=global_elements,
                         data_space=data_space,
                         guide_hidden_dim=guide_hidden_dim)
        self._strict_load = False
        self._amortized = amortized
        self._data_length = dataset_length

        if self._amortized:
            prior_logits_dict = pnn.PyroModule[nn.ModuleDict]()
        else:
            prior_logits_dict = pnn.PyroModule[nn.ParameterDict]()
        self.asvi_params = pnn.PyroModule[nn.ModuleDict]({
            'mean_fields': pnn.PyroModule[nn.ModuleDict](),
            'prior_logits': prior_logits_dict
        })

    def resume_from_checkpoint(self, resume_path):
        checkpoint = super().resume_from_checkpoint(resume_path, True)
        state_dict = checkpoint['state_dict']

        logits_prefix = 'asvi_params.prior_logits'
        saved_rvs = {k[len(logits_prefix)+1:].split('.')[0]
                     for k, v in state_dict.items() if logits_prefix in k}

        for name in saved_rvs:
            logits_qual = logits_prefix + '.' + name
            mean_fields_qual = 'asvi_params.mean_fields.' + name
            mean_fields = {k[len(mean_fields_qual) + 1:]: v
                           for k, v in state_dict.items()
                           if mean_fields_qual in k}

            if self._amortized:
                logits_extra = state_dict[logits_qual + '._extra_state']
                logits_module = getattr(modules, logits_extra['type'])
                self.asvi_params.prior_logits[name] = logits_module(
                    logits_extra['in'], logits_extra['out']
                )
                self.asvi_params.prior_logits[name].load_state_dict({
                    k[len(logits_qual)+1:]: v
                    for k, v in state_dict.items() if logits_qual in k
                })

                self.asvi_params.mean_fields[name] =\
                    pnn.PyroModule[nn.ModuleDict]()
                params = {k.split('.')[0] for k in mean_fields
                          if '_extra_state' in k}
                for key in params:
                    extra_qual = mean_fields_qual + '.' + key + '._extra_state'
                    extra = state_dict[extra_qual]
                    module = getattr(modules, extra['type'])
                    self.asvi_params.mean_fields[name][key] = module(
                        extra['in'], extra['out']
                    )
                    self.asvi_params.mean_fields[name][key].load_state_dict({
                        k[len(key)+1:]: v for k, v in mean_fields.items()
                        if key in k
                    })
            else:
                val = state_dict[logits_qual]
                self.asvi_params.prior_logits[name] = nn.Parameter(val.cpu())

                self.asvi_params.mean_fields[name] =\
                        pnn.PyroModule[nn.ParameterDict]()
                for k, v in mean_fields.items():
                    self.asvi_params.mean_fields[name][k] =\
                        nn.Parameter(v.cpu())

    @pnn.pyro_method
    def model(self, observations=None, valid=False, index=None, **kwargs):
        morphism, observations, data = super().model(observations)

        if observations is not None:
            score_morphism = pyro.condition(morphism, data=observations)
        else:
            score_morphism = morphism
        with pyro.plate('data', len(data)):
            with name_count():
                output = score_morphism()
        return morphism, output

    @pnn.pyro_method
    def guide(self, observations=None, valid=False, index=None, **kwargs):
        if isinstance(observations, dict):
            data = observations[self._observation_name]
        else:
            data = observations
        data = data.view(data.shape[0], *self._data_space)
        morphism, data = super().guide(observations)

        with pyro.plate('data', len(data)):
            with name_count():
                morphism = block(morphism, hide=[self._observation_name])
                if self._amortized:
                    morphism = amortized_asvi(morphism, self.asvi_params,
                                              ConvIncoder, data)
                else:
                    morphism = asvi(morphism, self.asvi_params, index=index,
                                    length=self._data_length)
                morphism()

        return morphism

class DeepGenerativeOperadicModel(AsviOperadicModel):
    def __init__(self, data_dim=28*28, hidden_dim=8, guide_hidden_dim=256,
                 **kwargs):
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
            data = {'effect': decoder.effect, 'function': decoder}
            generator = cart_closed.Box(decoder.name, decoder.type.left,
                                        decoder.type.right, data=data)
            generators.append(generator)

        obs = set()
        for generator in generators:
            ty = generator.dom >> generator.cod
            obs = obs | unification.base_elements(ty)

        global_elements = []
        no_prior_dims = [self._data_dim]
        for ob in obs:
            dim = types.type_size(str(ob))
            if dim in no_prior_dims:
                continue

            space = types.tensor_type(torch.float, dim)
            prior = StandardNormal(dim)
            name = '$p(%s)$' % prior.effects
            effect = {'effect': prior.effect, 'dagger_effect': [],
                      'function': prior}
            global_element = cart_closed.Box(name, Ty(), space, data=effect)
            global_elements.append(global_element)

        super().__init__(generators, global_elements, data_dim,
                         guide_hidden_dim, **kwargs)

class MolecularVaeOperadicModel(DaggerOperadicModel):
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
                        'dagger_effect': encoder.effect,
                        'function': decoder}
                conv_generator = cart_closed.Box(decoder.name,
                                                 decoder.type.left,
                                                 decoder.type.right, data=data)
                generators.append(conv_generator)
                data = {'dagger_effect': decoder.effect,
                        'effect': encoder.effect,
                        'function': encoder}
                conv_dagger = cart_closed.Box(encoder.name, encoder.type.left,
                                              encoder.type.right, data=data)
                dagger_generators.append(conv_dagger)

                encoder = RecurrentMolecularEncoder(hidden, recurrent,
                                                    charset_len, max_len)
                decoder = MolecularDecoder(hidden, recurrent_dim=recurrent,
                                           charset_len=charset_len,
                                           max_len=max_len)
                data = {'effect': decoder.effect,
                        'dagger_effect': encoder.effect,
                        'function': decoder}
                rec_generator = cart_closed.Box(decoder.name, decoder.type.left,
                                                decoder.type.right, data=data)
                generators.append(rec_generator)
                data = {'dagger_effect': decoder.effect,
                        'effect': encoder.effect,
                        'function': encoder}
                rec_dagger = cart_closed.Box(encoder.name, encoder.type.left,
                                             encoder.type.right, data=data)
                dagger_generators.append(rec_dagger)

        super().__init__(generators, [], data_space=(max_len, charset_len),
                         guide_hidden_dim=guide_hidden_dim,
                         no_prior_dims=[max_len, charset_len],
                         dagger_generators=dagger_generators)
