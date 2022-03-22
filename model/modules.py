from abc import abstractproperty
from discopy.biclosed import Ty
import functools
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
        return (dim_space @ dim_space) >> dim_space

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
        return Ty() >> types.tensor_type(torch.float, self._dim)

    def forward(self):
        z_loc = self._batch.new_zeros(torch.Size((self._batch.shape[0],
                                                  self._dim)))
        z_scale = self._batch.new_ones(torch.Size((self._batch.shape[0],
                                                   self._dim)))
        normal = dist.Normal(z_loc, z_scale).to_event(1)
        return pyro.sample('$%s$' % self._latent_name, normal)

class NullPrior(TypedModel):
    def __init__(self, dim, random_var_name=None):
        super().__init__()
        self._dim = dim
        if not random_var_name:
            random_var_name = 'X^{%d}' % self._dim[0]
        self._random_var_name = random_var_name

    @property
    def effect(self):
        return [self._random_var_name]

    @property
    def type(self):
        return Ty() >> types.tensor_type(torch.float, self._dim)

    def forward(self):
        size = torch.Size((self._batch.shape[0], self._dim))
        probs = self._batch.new_zeros(size)
        temps = self._batch.new_ones(size)
        bernoulli = dist.RelaxedBernoulli(temps, probs=probs).to_event(1)
        return pyro.sample('$%s$' % self._random_var_name, bernoulli)

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
        return types.tensor_type(torch.float, self._dim) >>\
               types.tensor_type(torch.float, self._dim)

    def forward(self, inputs):
        xs = torch.clamp(inputs.view(-1, self._dim[0]), 0., 1.)
        bernoulli = dist.ContinuousBernoulli(probs=xs).to_event(1)
        sample = pyro.sample('$%s$' % self._random_var_name, bernoulli)
        if self._likelihood:
            return xs
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
        return self._in_space >> self._out_space

    @property
    def effect(self):
        return self.distribution.effect

    @abstractproperty
    def density_name(self):
        raise NotImplementedError()

class DensityDecoder(DensityNet):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel,
                 convolve=False):
        super().__init__(in_dim, out_dim, dist_layer, convolve=convolve)

    @property
    def density_name(self):
        condition_name = 'Z^{%d}' % self._in_dim
        return '$p(%s | %s)$' % (self.effects, condition_name)

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

class DensityEncoder(DensityNet):
    def __init__(self, in_dim, out_dim, dist_layer=DiagonalGaussian,
                 convolve=False):
        super().__init__(in_dim, out_dim, dist_layer, convolve=convolve)

    @property
    def density_name(self):
        condition_name = 'Z^{%d}' % self._in_dim
        return '$q(%s | %s)$' % (self.effects, condition_name)

    def forward(self, inputs):
        if self._convolve:
            in_side = int(np.sqrt(self._in_dim))
            inputs = inputs.view(-1, 1, in_side, in_side)
            out_hidden = self.conv_layers(inputs)
            out_hidden = out_hidden.view(inputs.shape[0], -1)
            out_hidden = self.dense_layers(out_hidden)
        else:
            out_hidden = self.neural_layers(inputs)
        if self._channels == 2:
            out_hidden = out_hidden.view(-1, 2, self._out_dim)
            return self.distribution(out_hidden[:, 0], out_hidden[:, 1])
        return self.distribution(out_hidden.view(-1, self._out_dim))

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
        return (input_space @ noise_space) >>\
               types.tensor_type(torch.float, self._out_dim)

    @property
    def effect(self):
        if self.has_distribution:
            return self.distribution.effect
        return []

    @property
    def name(self):
        args_name = '(\\mathbb{R}^{%d} \\times \\mathbb{R}^{%d})'
        args_name = args_name % (self._in_dim, self._noise_dim)
        name = 'p(%s \\mid %s)' % (self.effects, args_name)
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
        return Ty() >> types.tensor_type(torch.float, self._out_dim)

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
        name = 'p(%s \\mid \\mathbb{R}^{%d})'
        name = name % (self.effects, self._noise_dim)
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

class LadderEncoder(TypedModel):
    def __init__(self, in_dim, out_dim, out_dist, noise_dist, noise_dim=2,
                 channels=1, conv=False):
        super().__init__()
        self._convolve = conv
        self._in_dim = in_dim
        self._noise_dim = noise_dim
        self._out_dim = out_dim
        self._num_channels = channels

        if out_dist is not None:
            self.ladder_distribution = out_dist(out_dim)
        out_features = out_dim
        if self.has_ladder_distribution and\
           isinstance(self.ladder_distribution, DiagonalGaussian):
            out_features *= 2
        self.noise_distribution = noise_dist(noise_dim)
        noise_features = noise_dim
        if isinstance(self.noise_distribution, DiagonalGaussian):
            noise_features *= 2

        if self._convolve:
            in_side = int(np.sqrt(self._in_dim))

            self.noise_convs = nn.Sequential(
                nn.Conv2d(self._num_channels, in_side, 4, 2, 1),
                nn.InstanceNorm2d(in_side), nn.PReLU(),
                nn.Conv2d(in_side, in_side * 2, 4, 2, 1),
                nn.InstanceNorm2d(in_side * 2), nn.PReLU(),
            )
            self.noise_linear = nn.Linear(in_side * 2 * (2 * 4 - 1) ** 2,
                                          noise_features)

            self.ladder_convs = nn.Sequential(
                nn.Conv2d(self._num_channels, in_side, 4, 2, 1),
                nn.InstanceNorm2d(in_side), nn.PReLU(),
                nn.Conv2d(in_side, in_side * 2, 4, 2, 1),
                nn.InstanceNorm2d(in_side * 2), nn.PReLU(),
            )
            self.ladder_linear = nn.Linear(in_side * 2 * (2 * 4 - 1) ** 2,
                                           out_features)
        else:
            self.noise_dense = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
                nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
                nn.Linear(in_dim, noise_features),
            )
            self.ladder_dense = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
                nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
                nn.Linear(in_dim, out_features),
            )

    @property
    def type(self):
        output_space = types.tensor_type(torch.float, self._out_dim)
        noise_space = types.tensor_type(torch.float, self._noise_dim)
        return types.tensor_type(torch.float, self._in_dim) >>\
               (output_space @ noise_space)

    @property
    def effect(self):
        effect = self.noise_distribution.effect
        if self.has_ladder_distribution:
            effect += self.ladder_distribution.effect
        return effect

    @property
    def name(self):
        name = 'q(%s \\mid %s)' % (self.effects,
                                   '\\mathbb{R}^{%d}' % self._in_dim)
        return '$%s$' % name

    @property
    def has_ladder_distribution(self):
        return hasattr(self, 'ladder_distribution')

    def forward(self, ladder_input):
        if self._convolve:
            in_side = int(np.sqrt(self._in_dim))
            ladder_input = ladder_input.reshape(-1, 1, in_side, in_side)

            noise = self.noise_linear(self.noise_convs(ladder_input).reshape(
                -1, in_side * 2 * (2 * 4 - 1) ** 2
            ))

            hiddens = self.ladder_convs(ladder_input).reshape(
                -1, in_side * 2 * (2 * 4 - 1) ** 2
            )
            hiddens = self.ladder_linear(hiddens)
        else:
            noise = self.noise_dense(ladder_input)
            hiddens = self.ladder_dense(ladder_input)

        noise = noise.view(-1, 2, self._noise_dim)
        noise = self.noise_distribution(noise[:, 0], noise[:, 1])

        if self.has_ladder_distribution:
            hiddens = hiddens.view(-1, 2, self._out_dim)
            hiddens = self.ladder_distribution(hiddens[:, 0], hiddens[:, 1])
        else:
            hiddens = hiddens.view(-1, self._out_dim)

        return hiddens, noise

class LadderPosterior(TypedModel):
    def __init__(self, in_dim, noise_dist):
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = in_dim // 2

        self.distribution = noise_dist(self._out_dim)
        noise_features = self._out_dim
        if isinstance(self.distribution, DiagonalGaussian):
            noise_features *= 2

        self.noise_dense = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
            nn.Linear(in_dim, noise_features),
        )

    @property
    def type(self):
        return types.tensor_type(torch.float, self._in_dim) >> Ty()

    @property
    def effect(self):
        return self.distribution.effect

    @property
    def name(self):
        name = 'q(%s \\mid %s)' % (self.effects,
                                   '\\mathbb{R}^{%d}' % self._in_dim)
        return '$%s$' % name

    def set_batching(self, batch):
        super().set_batching(batch)
        self.distribution.set_batching(batch)

    def forward(self, ladder_input):
        noise = self.noise_dense(ladder_input).view(-1, 2, self._out_dim)
        self.distribution(noise[:, 0], noise[:, 1])
        return ()

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
        return glimpse_type >> canvas_type

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

        return (canvas_type @ glimpse_type) >> canvas_type

    @property
    def effect(self):
        return self.coordinates_dist.effect

    @property
    def name(self):
        canvas_name = 'Z^{%d}' % self._canvas_side ** 2
        glimpse_name = 'Z^{%d}' % self._glimpse_side ** 2
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

class CanvasEncoder(TypedModel):
    def __init__(self, canvas_side=28, glimpse_side=7):
        super().__init__()
        self._canvas_side = canvas_side
        self._glimpse_side = glimpse_side

        glimpse_name = 'Z^{%d}' % glimpse_side ** 2
        self.glimpse_dist = DiagonalGaussian(self._glimpse_side ** 2,
                                             latent_name=glimpse_name)
        self.glimpse_precision = nn.Sequential(
            nn.Linear(self._canvas_side ** 2, self._glimpse_side ** 2),
            nn.Softplus(),
        )

    @property
    def type(self):
        canvas_type = types.tensor_type(torch.float, self._canvas_side ** 2)
        glimpse_type = types.tensor_type(torch.float, self._glimpse_side ** 2)

        return canvas_type >> glimpse_type

    @property
    def effect(self):
        return self.glimpse_dist.effect

    @property
    def name(self):
        canvas_name = 'X^{%d}' % self._canvas_side ** 2
        glimpse_name = 'Z^{%d}' % self._glimpse_side ** 2
        name = 'q(%s \\mid %s)' % (glimpse_name, canvas_name)
        return '$%s$' % name

    def canvas_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._canvas_side,
                           self._canvas_side])

    def glimpse_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._glimpse_side,
                           self._glimpse_side])

    def forward(self, canvas):
        glimpse_precision = self.glimpse_precision(canvas)
        canvas = canvas.view(*self.canvas_shape(canvas))

        coords = torch.tensor([1., 0., 0.]).to(canvas).expand(canvas.shape[0],
                                                              3)
        transforms = glimpse_transform(inverse_glimpse(coords))
        grid = F.affine_grid(transforms, self.glimpse_shape(canvas),
                             align_corners=True)
        glimpse = F.grid_sample(canvas, grid, align_corners=True)
        flat_glimpse = glimpse.view(-1, self._glimpse_side ** 2)

        glimpse = self.glimpse_dist(flat_glimpse, glimpse_precision)
        return glimpse

class SpatialTransformerReader(TypedModel):
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

        return canvas_type >> (canvas_type @ glimpse_type)

    @property
    def effect(self):
        return self.coordinates_dist.effect

    @property
    def name(self):
        canvas_name = 'Z^{%d}' % self._canvas_side ** 2
        glimpse_name = 'Z^{%d}' % self._glimpse_side ** 2
        outputs_tuple = ' \\times '.join([canvas_name, glimpse_name])
        name = 'q(%s \\mid %s)' % (outputs_tuple, canvas_name)
        return '$%s$' % name

    def canvas_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._canvas_side,
                           self._canvas_side])

    def glimpse_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._glimpse_side,
                           self._glimpse_side])

    def forward(self, images):
        images = images.view(*self.canvas_shape(images))

        coords = self.glimpse_conv(images)
        coords = self.glimpse_selector(coords).sum(dim=1)
        coords = self.glimpse_dense(
            coords.view(-1, (self._canvas_side // (2 ** 3)) ** 2)
        ).view(-1, 2, 3)
        coords = self.coordinates_dist(coords[:, 0], coords[:, 1])
        coords = torch.cat((F.softplus(coords[:, :1]), coords[:, 1:]), dim=-1)
        transforms = glimpse_transform(inverse_glimpse(coords))

        grid = F.affine_grid(transforms, self.glimpse_shape(images),
                             align_corners=True)
        glimpse = F.grid_sample(images, grid, align_corners=True)

        recon_transforms = glimpse_transform(coords)
        recon_grid = F.affine_grid(recon_transforms, self.canvas_shape(images),
                                   align_corners=True)
        glimpse_recon = F.grid_sample(glimpse, recon_grid, align_corners=True)

        residual = images - glimpse_recon

        return residual, glimpse

class RecurrentMolecularEncoder(TypedModel):
    def __init__(self, hidden_dim=196, recurrent_dim=488, charset_len=34,
                 max_len=120):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._charset_len = charset_len
        self._max_len = max_len

        self.encoder_linear = nn.Sequential(
            nn.Linear(self._charset_len, recurrent_dim),
            nn.SELU(),
        )
        self.recurrence = nn.GRU(recurrent_dim, hidden_dim, 3, batch_first=True)
        self.postrecurrence_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )
        self.embedding_loc = nn.Linear(hidden_dim, hidden_dim)
        self.embedding_log_scale = nn.Linear(hidden_dim, hidden_dim)
        self.embedding_dist = DiagonalGaussian(hidden_dim)

    @property
    def type(self):
        smiles_type = types.tensor_type(torch.float,
                                        (self._max_len, self._charset_len))
        embedding_type = types.tensor_type(torch.float, self._hidden_dim)
        return smiles_type >> embedding_type

    @property
    def effect(self):
        return self.embedding_dist.effect

    @property
    def name(self):
        smiles_name = 'X^{(%d, %d)}' % (self._max_len, self._charset_len)
        embedding_name = 'Z^{%d}' % self._hidden_dim
        name = 'q(%s \\mid %s)' % (embedding_name, smiles_name)
        return '$%s$' % name

    def forward(self, smiles):
        smiles = smiles.view(-1, self._max_len, self._charset_len)
        features, _ = self.recurrence(self.encoder_linear(smiles))
        features = self.postrecurrence_linear(features[:, 0])

        loc = self.embedding_loc(features)
        precision = (-self.embedding_log_scale(features)).exp()
        return self.embedding_dist(loc, precision)


class ConvMolecularEncoder(TypedModel):
    def __init__(self, hidden_dim=196, charset_len=34, max_len=120):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._charset_len = charset_len
        self._max_len = max_len

        self.smiles_conv = nn.Sequential(
            nn.Conv1d(120, 9, kernel_size=9), nn.ReLU(),
            nn.Conv1d(9, 9, kernel_size=9), nn.ReLU(),
            nn.Conv1d(9, 10, kernel_size=11), nn.ReLU(),
        )
        self.smiles_linear = nn.Sequential(
            nn.Linear(80, hidden_dim),
            nn.SELU(),
        )
        self.embedding_loc = nn.Linear(hidden_dim, hidden_dim)
        self.embedding_log_scale = nn.Linear(hidden_dim, hidden_dim)
        self.embedding_dist = DiagonalGaussian(hidden_dim)

    @property
    def type(self):
        smiles_type = types.tensor_type(torch.float,
                                        (self._max_len, self._charset_len))
        embedding_type = types.tensor_type(torch.float, self._hidden_dim)
        return smiles_type >> embedding_type

    @property
    def effect(self):
        return self.embedding_dist.effect

    @property
    def name(self):
        smiles_name = 'X^{(%d, %d)}' % (self._max_len, self._charset_len)
        embedding_name = 'Z^{%d}' % self._hidden_dim
        name = 'q(%s \\mid %s)' % (embedding_name, smiles_name)
        return '$%s$' % name

    def forward(self, smiles):
        smiles = smiles.view(-1, self._max_len, self._charset_len)
        features = self.smiles_conv(smiles).flatten(start_dim=1)
        features = self.smiles_linear(features)

        loc = self.embedding_loc(features)
        precision = (-self.embedding_log_scale(features)).exp()

        return self.embedding_dist(loc, precision)

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
        return embedding_type >> smiles_type

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

class RecurrentEncoder(TypedModel):
    def __init__(self, in_dims, out_dims, effects, hidden_dim=128):
        super().__init__()
        self._effects = effects
        self._eff_dims = [types.type_size(eff) for eff in self._effects]
        self._in_dims = in_dims
        self._out_dims = out_dims

        max_dim = max(*self._eff_dims, sum(self._in_dims), sum(self._out_dims),
                      hidden_dim) * 2

        self.recurrent = nn.GRUCell(sum(self._in_dims), max_dim)
        if sum(self._out_dims):
            self.next_encoder = nn.Sequential(
                nn.Linear(max_dim, max_dim),
                nn.PReLU(),
                nn.Linear(max_dim, sum(self._out_dims)),
            )

    @property
    def type(self):
        in_tys = [types.tensor_type(torch.float, in_dim) for in_dim
                  in self._in_dims]
        in_space = functools.reduce(lambda t, u: t @ u, in_tys, Ty())
        out_tys = [types.tensor_type(torch.float, out_dim) for out_dim
                   in self._out_dims]
        out_space = functools.reduce(lambda t, u: t @ u, out_tys, Ty())
        return in_space >> out_space

    @property
    def effect(self):
        return self._effects

    @property
    def name(self):
        data_name = 'X^{%d}' % self._in_dim
        name = 'p(%s \\mid %s)' % (self._effects.join(','), data_name)
        return '$%s$' % name

    def forward(self, *args):
        xs = torch.cat(args, dim=-1)

        hs = self.recurrent(xs)
        for i, effect in enumerate(self._effects):
            eff_dim = self._eff_dims[i]
            hs = self.recurrent(xs, hs)

            loc, scale = hs[:, :eff_dim], F.softplus(hs[:, eff_dim:eff_dim*2])
            normal = dist.Normal(loc, scale).to_event(1)
            zs = pyro.sample('$%s$' % effect, normal)
            hs = torch.cat((zs, hs[:, eff_dim:]), dim=-1)

        result = ()
        if self._out_dims:
            hs = self.next_encoder(hs)

            d = 0
            for dim in self._out_dims:
                result = result + (hs[:, d:d+dim],)
                d += dim

        return result
