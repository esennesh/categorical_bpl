from abc import abstractproperty
from discopy.monoidal import Ty
import functools
import math
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
from pyro.poutine.condition_messenger import ConditionMessenger
import pyro.poutine.runtime as runtime
import torch.distributions
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F

from base import TypedModel
import base.base_type as types
import utils.util as util

class DiagonalGaussian(TypedModel):
    def __init__(self, dim, latent_name=None, likelihood=False):
        super().__init__()
        self._dim = torch.Size([dim])
        if not latent_name:
            latent_name = 'Z^{%d}' % self._dim[0]
        self._latent_name = latent_name
        self._likelihood = likelihood

    @property
    def effect(self):
        return [self._latent_name]

    @property
    def type(self):
        dim_space = types.tensor_type(torch.float, self._dim)
        return (dim_space @ dim_space, dim_space)

    def forward(self, loc, precision):
        scale = torch.sqrt(F.softplus(precision)) ** (-1.)
        normal = dist.Normal(loc, scale).to_event(1)
        zs = pyro.sample('$%s$' % self._latent_name, normal)
        if self._likelihood:
            return loc
        return zs

class StandardNormal(TypedModel):
    def __init__(self, dim, latent_name=None):
        super().__init__()
        if not latent_name:
            latent_name = 'Z^{%d}' % dim
        self._latent_name = latent_name
        self._dim = dim

    @property
    def effect(self):
        return [self._latent_name]

    @property
    def type(self):
        return (Ty(), types.tensor_type(torch.float, self._dim))

    def forward(self):
        z_loc = self._batch.new_zeros(torch.Size((self._batch.shape[0],
                                                  self._dim)))
        z_scale = self._batch.new_ones(torch.Size((self._batch.shape[0],
                                                   self._dim)))
        normal = dist.Normal(z_loc, z_scale).to_event(1)
        return pyro.sample('$%s$' % self._latent_name, normal)

class NullPrior(TypedModel):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def effect(self):
        return []

    @property
    def type(self):
        return (Ty(), types.tensor_type(torch.float, self._dim))

    @property
    def name(self):
        name = '0: \\mathbb{R}^{%d}' % self._dim
        return '$%s$' % name

    def forward(self):
        size = torch.Size((self._batch.shape[0], self._dim))
        return self._batch.new_zeros(size)

class ContinuousBernoulliModel(TypedModel):
    def __init__(self, dim, random_var_name=None, likelihood=True):
        super().__init__()
        self._dim = torch.Size([dim])
        if not random_var_name:
            random_var_name = 'X^{%d}' % self._dim[0]
        self._random_var_name = random_var_name
        self._likelihood = likelihood

    @property
    def effect(self):
        return [self._random_var_name]

    @property
    def type(self):
        return (types.tensor_type(torch.float, self._dim),
                types.tensor_type(torch.float, self._dim))

    @property
    def name(self):
        name = 'p(%s \\mid \\mathbb{R}^{%d})'
        name = name % (self._random_var_name, self._dim[0])
        return '$%s$' % name

    def forward(self, inputs):
        xs = torch.clamp(inputs.view(-1, self._dim[0]), 0., 1.)
        bernoulli = dist.ContinuousBernoulli(probs=xs).to_event(1)
        sample = pyro.sample('$%s$' % self._random_var_name, bernoulli)
        if self._likelihood:
            return xs
        return sample

class GaussianLikelihood(DiagonalGaussian):
    def __init__(self, dim, random_var_name=None):
        super().__init__(dim, random_var_name, likelihood=True)
        self.precision = pnn.PyroParam(torch.ones(1),
                                       constraint=constraints.positive)

    @property
    def type(self):
        dim_space = types.tensor_type(torch.float, self._dim)
        return (dim_space, dim_space)

    @property
    def name(self):
        name = 'p(%s \\mid \\mathbb{R}^{%d})'
        name = name % (self._latent_name, self._dim[0])
        return '$%s$' % name

    def forward(self, loc):
        return super().forward(loc, self.precision.expand(*loc.shape))

class RelaxedOneHotLikelihood(TypedModel):
    def __init__(self, symbol, charset_len, max_len, likelihood=True):
        super().__init__()
        self._cod = types.tensor_type(torch.float, (max_len, charset_len))
        self._likelihood = likelihood
        self._max_len = max_len
        self._name = 'XC^{(%d, %d)}' % (max_len, charset_len)
        self._symbol = symbol
        self.temperature = pnn.PyroParam(torch.tensor([2.2]),
                                         constraint=constraints.positive)

    @property
    def type(self):
        return (Ty(self._symbol), self._cod)

    @property
    def effect(self):
        return [self._name]

    @property
    def name(self):
        name = 'p(%s \\mid %s)' % (self._name, 'Z' + self._name[1:])
        return '$%s$' % name

    def null_terminator(self, string):
        remainder = self._max_len - string.shape[1]
        charset_len = string.shape[-1]
        nil = torch.LongTensor([charset_len - 1])
        nil = F.one_hot(nil.to(device=string.device), charset_len)
        return nil.unsqueeze(dim=0).expand(string.shape[0], remainder,
                                           charset_len)

    def forward(self, string):
        # Zero-pad the incoming string
        if string.shape[1] < self._max_len:
            nil = self.null_terminator(string)
            string = torch.cat((string, nil), dim=1)
        onehot_cat = dist.RelaxedOneHotCategorical(self.temperature,
                                                   probs=string)
        sample = pyro.sample('$%s$' % self._name, onehot_cat.to_event(1))
        if self._likelihood:
            return string
        return sample

class DensityNet(TypedModel):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel,
                 normalizer_layer=nn.LayerNorm, convolve=False):
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._in_space = types.tensor_type(torch.float, in_dim)
        self._out_space = types.tensor_type(torch.float, out_dim)
        self._convolve = convolve

        self.add_module('distribution', dist_layer(out_dim))

        hidden_dim = (in_dim + out_dim) // 2
        self._channels = 1
        if isinstance(self.distribution, DiagonalGaussian):
            self._channels *= 2
        final_features = out_dim * self._channels
        if not self._convolve:
            self.add_module('neural_layers', nn.Sequential(
                nn.Linear(in_dim, hidden_dim), normalizer_layer(hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim), normalizer_layer(hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, final_features),
            ))
        else:
            if out_dim > in_dim:
                out_side = int(np.sqrt(self._out_dim))
                conv_side = max(out_side // 4, 1)
                multiplier = conv_side ** 2
                self.dense_layers = nn.Sequential(
                    nn.Linear(self._in_dim, multiplier * 2 * out_side),
                    normalizer_layer(multiplier * 2 * out_side), nn.PReLU(),
                    nn.Linear(multiplier * 2 * out_side,
                              multiplier * 2 * out_side),
                )
                self.conv_layers = nn.Sequential(
                    nn.ConvTranspose2d(2 * out_side, out_side, 4, 2, 1),
                    nn.InstanceNorm2d(out_side), nn.PReLU(),
                    nn.ConvTranspose2d(out_side, self._channels, 4, 2, 1),
                    nn.ReflectionPad2d((out_side - conv_side * 4) // 2)
                )
            else:
                in_side = int(np.sqrt(self._in_dim))
                multiplier = max(in_side // 4, 1) ** 2
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, in_side, 4, 2, 1),
                    nn.InstanceNorm2d(in_side), nn.PReLU(),
                    nn.Conv2d(in_side, in_side * 2, 4, 2, 1),
                    nn.InstanceNorm2d(in_side * 2), nn.PReLU(),
                )
                self.dense_layers = nn.Sequential(
                    nn.Linear(in_side * 2 * multiplier, final_features),
                    normalizer_layer(final_features), nn.PReLU(),
                    nn.Linear(final_features, final_features)
                )

    def set_batching(self, batch):
        super().set_batching(batch)
        self.distribution.set_batching(batch)

    @property
    def type(self):
        return (self._in_space, self._out_space)

    @property
    def effect(self):
        return self.distribution.effect

class DensityDecoder(DensityNet):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel,
                 convolve=False):
        super().__init__(in_dim, out_dim, dist_layer, convolve=convolve)

    @property
    def name(self):
        condition_name = '\\mathbb{R}^{%d}' % self._in_dim
        return '$p(%s \\mid %s)$' % (self.effects, condition_name)

    def forward(self, inputs):
        if self._convolve:
            hidden = self.dense_layers(inputs)
            out_side = int(np.sqrt(self._out_dim))
            conv_side = max(out_side // 4, 1)
            hidden = hidden.view(hidden.shape[0], 2 * out_side, conv_side,
                                 conv_side)
            hidden = self.conv_layers(hidden)
        else:
            hidden = self.neural_layers(inputs)
        if self._channels == 1:
            hidden = hidden.squeeze(dim=1).view(hidden.shape[0], self._out_dim)
            result = self.distribution(hidden)
        elif self._channels == 2:
            hidden = hidden.view(hidden.shape[0], 2, self._out_dim)
            result = self.distribution(hidden[:, 0], hidden[:, 1])
        return result

class LadderDecoder(TypedModel):
    def __init__(self, in_dim, out_dim, out_dist, noise_dim=2, channels=1,
                 conv=False):
        super().__init__()
        self._convolve = conv
        self._in_dim = in_dim
        self._noise_dim = noise_dim
        self._out_dim = out_dim
        self._num_channels = channels

        if out_dist is not None:
            self.distribution = out_dist(out_dim)
        final_features = out_dim
        if self.has_distribution and\
           isinstance(self.distribution, DiagonalGaussian):
            final_features *= 2

        self.noise_layer = nn.Sequential(nn.Linear(self._noise_dim, in_dim),
                                         nn.LayerNorm(in_dim), nn.PReLU())
        if self._convolve:
            out_side = int(np.sqrt(self._out_dim))
            self._multiplier = max(out_side // 4, 1) ** 2
            channels = self._num_channels
            if self.has_distribution and\
               isinstance(self.distribution, DiagonalGaussian):
                channels *= 2
            self.dense_layers = nn.Sequential(
                nn.Linear(self._in_dim * 2, self._multiplier * 2 * out_side),
                nn.LayerNorm(self._multiplier * 2 * out_side), nn.PReLU(),
            )
            self.conv_layers = nn.Sequential(
                nn.ConvTranspose2d(2 * out_side, out_side, 4, 2, 1),
                nn.InstanceNorm2d(out_side), nn.PReLU(),
                nn.ConvTranspose2d(out_side, channels, 4, 2, 1),
            )
        else:
            self.neural_layers = nn.Sequential(
                nn.Linear(self._in_dim * 2, self._out_dim),
                nn.LayerNorm(self._out_dim), nn.PReLU(),
                nn.Linear(self._out_dim, self._out_dim),
                nn.LayerNorm(self._out_dim), nn.PReLU(),
                nn.Linear(self._out_dim, self._out_dim),
                nn.LayerNorm(self._out_dim), nn.PReLU(),
                nn.Linear(self._out_dim, final_features)
            )

    @property
    def type(self):
        input_space = types.tensor_type(torch.float, self._in_dim)
        noise_space = types.tensor_type(torch.float, self._noise_dim)
        return (input_space @ noise_space,
                types.tensor_type(torch.float, self._out_dim))

    @property
    def effect(self):
        if self.has_distribution:
            return self.distribution.effect
        return []

    @property
    def name(self):
        args_name = '\\mathbb{R}^{%d} \\times \\mathbb{R}^{%d}'
        args_name = args_name % (self._in_dim, self._noise_dim)
        name = ''
        if self.effects:
            name = 'p(%s \\mid %s):' % (self.effects, args_name)
        name = name + '%s -> \\mathbb{R}^{%d}' % (args_name, self._out_dim)
        return '$%s$' % name

    @property
    def has_distribution(self):
        return hasattr(self, 'distribution')

    def forward(self, ladder_input, noise):
        hiddens = torch.cat((ladder_input, self.noise_layer(noise)), dim=-1)
        if self._convolve:
            multiplier = int(np.sqrt(self._multiplier))
            out_side = int(np.sqrt(self._out_dim))
            hiddens = self.dense_layers(hiddens).reshape(-1, out_side * 2,
                                                         multiplier,
                                                         multiplier)
            hiddens = self.conv_layers(hiddens).reshape(-1, out_side,
                                                        out_side)
        else:
            hiddens = self.neural_layers(hiddens)

        if self.has_distribution:
            if isinstance(self.distribution, ContinuousBernoulliModel):
                hiddens = self.distribution(hiddens)
            else:
                hiddens = hiddens.view(-1, 2, self._out_dim)
                hiddens = self.distribution(hiddens[:, 0], hiddens[:, 1])
        else:
            hiddens = hiddens.view(-1, self._out_dim)
        return hiddens

class LadderPrior(TypedModel):
    def __init__(self, out_dim, out_dist=DiagonalGaussian, channels=1):
        super().__init__()
        self._noise_dim = out_dim // 2
        self._out_dim = out_dim
        self._num_channels = channels

        self.noise_distribution = StandardNormal(self._noise_dim)

        if out_dist is not None:
            self.distribution = out_dist(out_dim)

        final_features = out_dim
        if self.has_distribution and\
           isinstance(self.distribution, DiagonalGaussian):
            final_features *= 2
        self.noise_dense = nn.Sequential(
            nn.Linear(self._noise_dim, self._out_dim),
            nn.LayerNorm(self._out_dim), nn.PReLU(),
            nn.Linear(self._out_dim, self._out_dim),
            nn.LayerNorm(self._out_dim), nn.PReLU(),
            nn.Linear(self._out_dim, self._out_dim),
            nn.LayerNorm(self._out_dim), nn.PReLU(),
            nn.Linear(self._out_dim, final_features)
        )

    @property
    def type(self):
        return (Ty(), types.tensor_type(torch.float, self._out_dim))

    @property
    def has_distribution(self):
        return hasattr(self, 'distribution')

    @property
    def effect(self):
        effect = self.noise_distribution.effect
        if self.has_distribution:
            effect += self.distribution.effect
        return effect

    @property
    def name(self):
        name = 'p(%s): Ty() -> \\mathbb{R}^{%d}' % (self.effects, self._out_dim)
        return '$%s$' % name

    def set_batching(self, batch):
        super().set_batching(batch)
        self.noise_distribution.set_batching(batch)
        if self.has_distribution:
            self.distribution.set_batching(batch)

    def forward(self):
        noise = self.noise_dense(self.noise_distribution())
        if self.has_distribution:
            noise = noise.view(-1, 2, self._out_dim)
            return self.distribution(noise[:, 0], noise[:, 1])
        return noise

def glimpse_transform(glimpse_code):
    scalings = torch.eye(2).expand(glimpse_code.shape[0], 2, 2).to(glimpse_code)
    scalings = scalings * glimpse_code[:, 0].view(-1, 1, 1)
    return torch.cat((scalings, glimpse_code[:, 1:].unsqueeze(-1)), dim=-1)

def inverse_glimpse(glimpse_code):
    coords = glimpse_code[:, 1:]
    scalars = glimpse_code[:, 0]
    return torch.cat((torch.ones(glimpse_code.shape[0], 1).to(coords), -coords),
                     dim=-1) / scalars.unsqueeze(-1)

class CanvasPrior(TypedModel):
    def __init__(self, canvas_side=28, glimpse_side=7):
        super().__init__()
        self._canvas_side = canvas_side
        self._glimpse_side = glimpse_side
        canvas_name = 'X^{%d}' % canvas_side ** 2
        self.distribution = DiagonalGaussian(self._canvas_side ** 2,
                                             latent_name=canvas_name)

        self.canvas_precision = nn.Sequential(
            nn.Linear(self._glimpse_side ** 2, self._canvas_side ** 2),
            nn.Softplus(),
        )

    @property
    def type(self):
        glimpse_type = types.tensor_type(torch.float, self._glimpse_side ** 2)
        canvas_type = types.tensor_type(torch.float, self._canvas_side ** 2)
        return (glimpse_type, canvas_type)

    @property
    def effect(self):
        return self.distribution.effect

    @property
    def name(self):
        glimpse_name = 'Z^{%d}' % self._glimpse_side ** 2
        name = 'p(%s \\mid %s)' % (self.distribution.effect[0],
                                   glimpse_name)
        return '$%s$' % name

    def canvas_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._canvas_side,
                           self._canvas_side])

    def glimpse_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._glimpse_side,
                           self._glimpse_side])

    def forward(self, glimpse):
        canvas_shape = self.canvas_shape(glimpse)
        glimpse_shape = self.glimpse_shape(glimpse)

        coords = torch.tensor([1., 0., 0.]).to(glimpse).expand(glimpse.shape[0],
                                                               3)
        glimpse_transforms = glimpse_transform(coords)
        grids = F.affine_grid(glimpse_transforms, canvas_shape,
                              align_corners=True)
        precision = self.canvas_precision(glimpse)
        glimpse = glimpse.view(glimpse_shape)
        flat_canvas = F.grid_sample(glimpse, grids, align_corners=True).view(
            -1, self._canvas_side ** 2
        )

        return self.distribution(flat_canvas, precision)

class SpatialTransformerWriter(TypedModel):
    def __init__(self, canvas_side=28, glimpse_side=7):
        super().__init__()
        self._canvas_side = canvas_side
        self._glimpse_side = glimpse_side

        self.glimpse_conv = nn.Sequential(
            nn.Conv2d(1, canvas_side, 4, 2, 1),
            nn.InstanceNorm2d(canvas_side), nn.PReLU(),
            nn.Conv2d(canvas_side, canvas_side * 2, 4, 2, 1),
            nn.InstanceNorm2d(canvas_side * 2), nn.PReLU(),
            nn.Conv2d(canvas_side * 2, canvas_side * 4, 4, 2, 1),
        )
        self.glimpse_selector = nn.Softmax2d()
        self.glimpse_dense = nn.Linear((self._canvas_side // (2 ** 3)) ** 2,
                                       3 * 2)
        self.coordinates_dist = DiagonalGaussian(3)

    @property
    def type(self):
        canvas_type = types.tensor_type(torch.float, self._canvas_side ** 2)
        glimpse_type = types.tensor_type(torch.float, self._glimpse_side ** 2)

        return (canvas_type @ glimpse_type, canvas_type)

    @property
    def effect(self):
        return self.coordinates_dist.effect

    @property
    def name(self):
        canvas_name = '\\mathbb{R}^{%d}' % self._canvas_side ** 2
        glimpse_name = '\\mathbb{R}^{%d}' % self._glimpse_side ** 2
        inputs_tuple = ' \\times '.join([canvas_name, glimpse_name])
        name = 'p(%s \\mid %s)' % (self.coordinates_dist.effect[0], inputs_tuple)
        return '$%s$' % name

    def canvas_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._canvas_side,
                           self._canvas_side])

    def glimpse_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._glimpse_side,
                           self._glimpse_side])

    def forward(self, canvas, glimpse_contents):
        canvas = canvas.view(*self.canvas_shape(canvas))

        coords = self.glimpse_conv(canvas)
        coords = self.glimpse_selector(coords).sum(dim=1)
        coords = self.glimpse_dense(
            coords.view(-1, (self._canvas_side // (2 ** 3)) ** 2)
        ).view(-1, 2, 3)
        coords = self.coordinates_dist(coords[:, 0], coords[:, 1])
        coords = torch.cat((F.softplus(coords[:, :1]), coords[:, 1:]), dim=-1)

        glimpse_transforms = glimpse_transform(coords)
        grids = F.affine_grid(glimpse_transforms, self.canvas_shape(canvas),
                              align_corners=True)
        glimpse_contents = glimpse_contents.view(*self.glimpse_shape(canvas))
        glimpse = F.grid_sample(glimpse_contents, grids, align_corners=True)

        canvas = canvas.flatten(1)
        glimpse = glimpse.flatten(1)
        return canvas + glimpse

class StringDecoder(TypedModel):
    def __init__(self, cod_nonterminal, const_productions, char_indices,
                 latent_space):
        super().__init__()
        self._char_indices = char_indices
        self._cod = Ty(cod_nonterminal)
        self._latent_space = latent_space
        self._productions = const_productions

        self.decoder = nn.Sequential(
            nn.Linear(math.prod(self._latent_space),
                      len(self._productions) * 2),
            nn.ReLU(),
            nn.Linear(len(self._productions) * 2, len(self._productions)),
            nn.Softplus(),
        )

    @property
    def type(self):
        return (types.tensor_type(torch.float, self._latent_space), self._cod)

    @property
    def _string_name(self):
        return 'C^{(%d, %s)}' % (len(self._char_indices), self._cod.name)

    @property
    def effect(self):
        return [self._string_name]

    @property
    def name(self):
        name = 'p(%s)' % self._string_name
        return '$%s$' % name

    def forward(self, zs):
        probs = self.decoder(zs)
        categorical = dist.Categorical(probs=probs)
        selection = pyro.sample('$%s$' % self._string_name, categorical)

        strings = []
        for choice in selection.unbind(dim=0):
            for token in self._productions[choice.item()]:
                chars = torch.LongTensor([self._char_indices[char] for char in
                                          token])
                chars = F.one_hot(chars.to(choice.device),
                                  len(self._char_indices))
                strings.append(chars)
        return torch.stack(strings, dim=0)


class ProductionDecoder(TypedModel):
    def __init__(self, cod_nonterminal, production, char_indices, max_len=120):
        super().__init__()
        self._production = production
        self._char_indices = char_indices
        self._cod = Ty(cod_nonterminal)
        self._dom = Ty(*[util.desymbolize(token) for token in production])

    @property
    def type(self):
        return (self._dom, self._cod)

    @property
    def _string_name(self):
        return 'C^{(%d, %s)}' % (len(self._char_indices), self._cod.name)

    @property
    def effect(self):
        return [self._string_name]

    @property
    def name(self):
        name = 'p(%s \\mid %s)' % (self._string_name, str(self._production))
        return '$%s$' % name

    def forward(self, *args):
        symbols = []
        i = 0
        for symbol in self._production:
            if isinstance(symbol, str):
                char = torch.LongTensor([self._char_indices[symbol]])
                char = F.one_hot(char.to(self._batch.device),
                                 len(self._char_indices))
                symbols.append(char.expand(self._batch.shape[0], *char.shape))
            else:
                symbols.append(args[i])
                i += 1
        return torch.cat(symbols, dim=1)

class MolecularDecoder(TypedModel):
    def __init__(self, hidden_dim=196, recurrent_dim=488, charset_len=34,
                 max_len=120):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._charset_len = charset_len
        self._max_len = max_len

        self.pre_recurrence_linear = nn.Sequential(
            nn.Linear(hidden_dim, self._charset_len),
            nn.SELU(),
        )
        self.recurrence1 = nn.GRUCell(self._charset_len, recurrent_dim)
        self.recurrence2 = nn.GRUCell(recurrent_dim, recurrent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(recurrent_dim, self._charset_len),
            nn.Softmax(dim=-1)
        )

    @property
    def _smiles_name(self):
        return 'X^{(%d, %d)}' % (self._max_len, self._charset_len)

    @property
    def type(self):
        embedding_type = types.tensor_type(torch.float, self._hidden_dim)
        smiles_type = types.tensor_type(torch.float,
                                        (self._max_len, self._charset_len))
        return (embedding_type, smiles_type)

    @property
    def effect(self):
        return [self._smiles_name]

    @property
    def name(self):
        embedding_name = 'Z^{%d}' % self._hidden_dim
        name = 'p(%s \\mid %s)' % (self._smiles_name, embedding_name)
        return '$%s$' % name

    def forward(self, zs):
        embedding = self.pre_recurrence_linear(zs)
        hiddens = [None, None]
        teacher = None
        if runtime.am_i_wrapped() and\
           isinstance(runtime._PYRO_STACK[-1], ConditionMessenger):
            data = runtime._PYRO_STACK[-1].data
            if '$%s$' % self._smiles_name in data:
                teacher = data['$%s$' % self._smiles_name]

        probs = []
        for i in range(self._max_len):
            hiddens[0] = self.recurrence1(embedding, hiddens[0])
            hiddens[1] = self.recurrence2(hiddens[0], hiddens[1])
            embedding = self.decoder(hiddens[1])

            probs.append(embedding)
            if teacher is not None:
                embedding = teacher[:, i]
        probs = torch.stack(probs, dim=1)

        logits_categorical = dist.OneHotCategorical(probs=probs).to_event(1)
        return pyro.sample('$%s$' % self._smiles_name, logits_categorical)

class Encoder(TypedModel):
    def __init__(self, in_dims, out_dims, latents, hidden_dim, incoder_cls,
                 normalizer_layer=nn.LayerNorm):
        super().__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self._z_dims = [types.type_size(latent) for latent in latents]
        self._effects = latents
        self._hidden_dim = hidden_dim

        self._incode = self._effects and sum(self._in_dims) != self._hidden_dim

        if self._incode:
            self.incoder = incoder_cls(sum(self._in_dims), self._hidden_dim,
                                       normalizer_layer)
        outcoder_dom = sum(self._in_dims) + sum(self._z_dims)

        outcoder_cod = sum(self._out_dims)
        if outcoder_cod:
            self.outcoder = nn.Sequential(
                normalizer_layer(outcoder_dom), nn.PReLU(),
                nn.Linear(outcoder_dom, outcoder_dom),
                normalizer_layer(outcoder_dom), nn.PReLU(),
                nn.Linear(outcoder_dom, outcoder_cod),
            )

    @property
    def type(self):
        in_tys = [types.tensor_type(torch.float, in_dim) for in_dim
                  in self._in_dims]
        in_space = functools.reduce(lambda t, u: t @ u, in_tys, Ty())
        out_tys = [types.tensor_type(torch.float, out_dim) for out_dim
                   in self._out_dims]
        out_space = functools.reduce(lambda t, u: t @ u, out_tys, Ty())
        return (in_space, out_space)

    @property
    def effect(self):
        return self._effects

    @property
    def name(self):
        in_names = ['\\mathbb{R}^{%d}' % dim for dim in self._in_dims]

        name = 'p(%s \\mid %s)' % (','.join(self.effect), ','.join(in_names))
        return '$%s$' % name

    def incode(self, xs):
        if self._incode:
            return self.incoder(xs)
        return xs

    def outcode(self, os, ins):
        result = ()
        if sum(self._z_dims):
            os = torch.cat((os, ins), dim=-1)

        if sum(self._out_dims):
            os = self.outcoder(os)
            d = 0
            for dim in self._out_dims:
                result = result + (os[:, d:d+dim],)
                d += dim
        return result

class Incoder(nn.Module):
    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    def get_extra_state(self):
        return {'type': self.__class__.__name__, 'in': self.in_features,
                'out': self.out_features}

    def set_extra_state(self, state):
        pass

class DenseIncoder(Incoder):
    def __init__(self, in_features, out_features, normalizer_cls=nn.LayerNorm):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_features, out_features),
            normalizer_cls(out_features), nn.PReLU(),
            nn.Linear(out_features, out_features),
        )
        self._in_features = in_features
        self._out_features = out_features


    def forward(self, features):
        return self.dense(features)

class ConvIncoder(Incoder):
    def __init__(self, in_features, out_features, normalizer_cls=nn.LayerNorm):
        super().__init__()
        self._in_side = int(np.sqrt(in_features))
        self._multiplier = max(self._in_side // 4, 1) ** 2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self._in_side, 4, 2, 1),
            nn.InstanceNorm2d(self._in_side), nn.PReLU(),
            nn.Conv2d(self._in_side, self._in_side * 2, 4, 2, 1),
            nn.InstanceNorm2d(self._in_side * 2), nn.PReLU(),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(self._in_side * 2 * self._multiplier, out_features),
            normalizer_cls(out_features), nn.PReLU(),
            nn.Linear(out_features, out_features)
        )
        self._in_features = in_features
        self._out_features = out_features

    def forward(self, features):
        features = features.reshape(-1, 1, self._in_side, self._in_side)
        hs = self.conv_layers(features)
        hs = hs.view(-1, self._in_side * 2 * self._multiplier)
        return self.dense_layers(hs)

class RecurrentEncoder(Encoder):
    def __init__(self, in_dims, out_dims, effects, incoder_cls=DenseIncoder):
        z_dims = [types.type_size(effect) for effect in effects]
        super().__init__(in_dims, out_dims, effects, max(z_dims) * 2,
                         incoder_cls)

        self.recurrent = nn.GRUCell(sum(self._z_dims), self._hidden_dim)

    @property
    def type(self):
        in_tys = [types.tensor_type(torch.float, in_dim) for in_dim
                  in self._in_dims]
        in_space = functools.reduce(lambda t, u: t @ u, in_tys, Ty())
        out_tys = [types.tensor_type(torch.float, out_dim) for out_dim
                   in self._out_dims]
        out_space = functools.reduce(lambda t, u: t @ u, out_tys, Ty())
        return (in_space, out_space)

    @property
    def effect(self):
        return self._effects

    @property
    def name(self):
        data_name = '\\mathbb{R}^{%d}' % self._in_dims
        name = 'p(%s \\mid %s)' % (self._effects.join(','), data_name)
        return '$%s$' % name

    def forward(self, *args):
        xs = torch.cat(args, dim=-1)
        hs = self.incode(xs)

        accumulated_zs = torch.zeros(hs.shape[0], sum(self._z_dims)).to(hs)
        d = 0
        for i, effect in enumerate(self._effects):
            z_dim = self._z_dims[i]
            hs = self.recurrent(accumulated_zs, hs)

            loc = hs[:, :z_dim]
            scale = F.softplus(hs[:, z_dim:z_dim*2])
            normal = dist.Normal(loc, scale).to_event(1)
            zs = pyro.sample('$%s$' % effect, normal)
            zs = torch.cat((accumulated_zs[:, :d], zs,
                            accumulated_zs[:, d+z_dim:]), dim=-1)
            d += z_dim
        if not self._effects:
            accumulated_zs = hs

        return self.outcode(accumulated_zs, xs)

class MlpEncoder(Encoder):
    def __init__(self, in_dims, out_dims, latent=None, incoder_cls=DenseIncoder,
                 normalizer_layer=nn.LayerNorm):
        hidden_dim = types.type_size(latent) * 2 if latent else sum(out_dims)
        super().__init__(in_dims, out_dims, [latent] if latent else [],
                         hidden_dim, incoder_cls,
                         normalizer_layer=normalizer_layer)
        if latent:
            self.distribution = DiagonalGaussian(self._hidden_dim,
                                                 latent_name=latent)

    def forward(self, *args):
        xs = torch.cat(args, dim=-1)
        zs = self.incode(xs)
        if self._effects:
            zs = zs.view(-1, 2, self._z_dims[0])
            loc, precision = zs[:, 0], F.softplus(zs[:, 1])
            zs = self.distribution(loc, precision)

        return self.outcode(zs, xs)

class StringEncoder(TypedModel):
    def __init__(self, effect, in_dims=(12,), out_features=20,
                 string_type='chars'):
        super().__init__()
        self._effect = effect
        self._in_dims = in_dims
        self._out_dim = out_features

        in_features = in_dims[0]
        if string_type == 'chars':
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_features, 9, kernel_size=9),
                nn.ReLU(),
                nn.Conv1d(9, 9, kernel_size=9),
                nn.ReLU(),
                nn.Conv1d(9, 10, kernel_size=11),
                nn.ReLU(),
            )
            self.dense = nn.Sequential(
                nn.Linear(80, 435), nn.ReLU(),
                nn.Linear(435, out_features * 2),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_features, 24, kernel_size=2),
                nn.ReLU(),
                nn.Conv1d(24, 12, kernel_size=3),
                nn.ReLU(),
                nn.Conv1d(12, 12, kernel_size=4),
                nn.ReLU(),
            )
            self.dense = nn.Sequential(nn.Linear(108, out_features), nn.ReLU())

        self.distribution = DiagonalGaussian(self._out_dim,
                                             latent_name=self._effect)

    @property
    def type(self):
        in_space = types.tensor_type(torch.float, self._in_dims)
        out_space = types.tensor_type(torch.float, self._out_dim)
        return (in_space, out_space)

    @property
    def name(self):
        name = 'p(%s \\mid \\mathbb{R}^{%s})' % (','.join(self._effect),
                                                 ','.join(self._in_dims))
        return '$%s$' % name

    def forward(self, features):
        hs = self.conv_layers(features).view(features.shape[0], -1)
        hs = self.dense(hs).view(features.shape[0], 2, self._out_dim)
        return self.distribution(hs[:, 0], hs[:, 1].exp())


def build_encoder(in_dims, out_dims, effects):
    latents = [eff for eff in effects if 'X^' not in eff]
    if len(in_dims) == 1 and (set(effects) - set(latents)):
        incoder_cls = ConvIncoder
    else:
        incoder_cls = DenseIncoder

    if len(latents) > 1:
        return RecurrentEncoder(in_dims, out_dims, latents, incoder_cls)
    return MlpEncoder(in_dims, out_dims, latents[0] if latents else None,
                      incoder_cls)
