from abc import abstractproperty
import collections
from discopy import Ty
from discopyro import cartesian_cat, closed
from indexed import IndexedOrderedDict
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.distributions
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, TypedModel
import base.base_type as types
import utils.util as util
from utils.name_stack import name_push, name_pop

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

    @property
    def random_var_name(self):
        return self._latent_name

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._dim * 2),
            types.tensor_type(torch.float, self._dim),
        )

    def forward(self, inputs):
        zs = inputs.view(-1, 2, self._dim[0])
        mean, std_dev = zs[:, 0], F.softplus(zs[:, 1])
        normal = dist.Normal(mean, std_dev).to_event(1)
        return pyro.sample('$%s$' % self._latent_name, normal)

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
            types.tensor_type(torch.float, self._dim),
        )

    def forward(self):
        z_loc = self._batch.new_zeros(torch.Size((self._batch.shape[0],
                                                  self._dim)))
        z_scale = self._batch.new_ones(torch.Size((self._batch.shape[0],
                                                   self._dim)))
        normal = dist.Normal(z_loc, z_scale).to_event(1)
        return pyro.sample('$%s$' % self._latent_name, normal)

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
        xs = torch.sigmoid(inputs.view(-1, self._obs_dim[0]))
        bernoulli = ContinuousBernoulli(probs=xs).to_event(1)
        pyro.sample('$%s$' % self._observable_name, bernoulli)
        return xs

class DensityNet(TypedModel):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel,
                 normalizer_layer=nn.LayerNorm):
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._in_space = types.tensor_type(torch.float, in_dim)
        self._out_space = types.tensor_type(torch.float, out_dim)

        hidden_dim = (in_dim + out_dim) // 2
        final_features = out_dim
        if dist_layer == DiagonalGaussian:
            final_features *= 2
        self.add_module('neural_layers', nn.Sequential(
            nn.Linear(in_dim, hidden_dim), normalizer_layer(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), normalizer_layer(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, final_features),
        ))
        self.add_module('distribution', dist_layer(out_dim))

    def set_batching(self, batch):
        super().set_batching(batch)
        self.distribution.set_batching(batch)

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(self._in_space, self._out_space)

    @abstractproperty
    def density_name(self):
        raise NotImplementedError()

class DensityDecoder(DensityNet):
    def __init__(self, in_dim, out_dim, dist_layer=ContinuousBernoulliModel):
        super().__init__(in_dim, out_dim, dist_layer)

    @property
    def density_name(self):
        sample_name = self.distribution.random_var_name
        condition_name = 'Z^{%d}' % self._in_dim
        return '$p(%s | %s)$' % (sample_name, condition_name)

    def forward(self, inputs):
        hidden = self.neural_layers(inputs)
        return self.distribution(hidden)

class DensityEncoder(DensityNet):
    def __init__(self, in_dim, out_dim, dist_layer=DiagonalGaussian):
        super().__init__(in_dim, out_dim, dist_layer)

    @property
    def density_name(self):
        sample_name = self.distribution.random_var_name
        condition_name = 'Z^{%d}' % self._in_dim
        return '$q(%s | %s)$' % (sample_name, condition_name)

    def forward(self, inputs):
        out_hidden = self.neural_layers(inputs)
        return self.distribution(out_hidden)

class LadderDecoder(TypedModel):
    def __init__(self, in_dim, out_dim, out_dist, noise_dim=2, channels=1,
                 conv=False):
        super().__init__()
        self._convolve = conv
        self._in_dim = in_dim
        self._noise_dim = noise_dim
        self._out_dim = out_dim
        self._num_channels = channels

        self.distribution = out_dist(out_dim)
        final_features = out_dim
        if out_dist == DiagonalGaussian:
            final_features *= 2

        self.noise_layer = nn.Sequential(nn.Linear(self._noise_dim, in_dim),
                                         nn.LayerNorm(in_dim), nn.PReLU())
        if self._convolve:
            out_side = int(np.sqrt(self._out_dim))
            self._multiplier = max(out_side // 4, 1) ** 2
            channels = self._num_channels
            if out_dist == DiagonalGaussian:
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
        return closed.CartesianClosed.ARROW(
            closed.CartesianClosed.BASE(Ty(input_space, noise_space)),
            types.tensor_type(torch.float, self._out_dim)
        )

    @property
    def name(self):
        args_name = '(\\mathbb{R}^{%d} \\times \\mathbb{R}^{%d})'
        args_name = args_name % (self._in_dim, self._noise_dim)
        name = 'p(%s \\mid %s)' % (self.distribution.random_var_name, args_name)
        return '$%s$' % name

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

        return self.distribution(hiddens)

class LadderPrior(TypedModel):
    def __init__(self, noise_dim, out_dim, out_dist=DiagonalGaussian,
                 channels=1):
        super().__init__()
        self._in_dim = noise_dim
        self._out_dim = out_dim
        self._num_channels = channels

        final_features = out_dim
        if out_dist == DiagonalGaussian:
            final_features *= 2
        self.noise_dense = nn.Sequential(
            nn.Linear(self._in_dim, self._out_dim),
            nn.LayerNorm(self._out_dim), nn.PReLU(),
            nn.Linear(self._out_dim, self._out_dim),
            nn.LayerNorm(self._out_dim), nn.PReLU(),
            nn.Linear(self._out_dim, self._out_dim),
            nn.LayerNorm(self._out_dim), nn.PReLU(),
            nn.Linear(self._out_dim, final_features)
        )
        self.distribution = out_dist(out_dim)

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._in_dim),
            types.tensor_type(torch.float, self._out_dim)
        )

    @property
    def name(self):
        name = 'p(%s \\mid \\mathbb{R}^{%d})'
        name = name % (self.distribution.random_var_name, self._in_dim)
        return '$%s$' % name

    def forward(self, noise):
        return self.distribution(self.noise_dense(noise))

class LadderEncoder(TypedModel):
    def __init__(self, in_dim, out_dim, out_dist, noise_dist, noise_dim=2,
                 channels=1, conv=False):
        super().__init__()
        self._convolve = conv
        self._in_dim = in_dim
        self._noise_dim = noise_dim
        self._out_dim = out_dim
        self._num_channels = channels

        self.ladder_distribution = out_dist(out_dim)
        out_features = out_dim
        if out_dist == DiagonalGaussian:
            out_features *= 2
        self.noise_distribution = noise_dist(noise_dim)
        noise_features = noise_dim
        if out_dist == DiagonalGaussian:
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
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._in_dim),
            closed.CartesianClosed.BASE(Ty(output_space, noise_space)),
        )

    @property
    def name(self):
        args_name = '(\\mathbb{R}^{%d} \\times \\mathbb{R}^{%d})'
        args_name = args_name % (self._out_dim, self._noise_dim)
        name = 'q(%s \\mid %s)' % (args_name, '\\mathbb{R}^{%d}' % self._in_dim)
        return '$%s$' % name

    def forward(self, ladder_input):
        if self._convolve:
            in_side = int(np.sqrt(self._in_dim))
            ladder_input = ladder_input.reshape(-1, 1, in_side, in_side)

            noise = self.noise_convs(ladder_input).reshape(
                -1, in_side * 2 * (2 * 4 - 1) ** 2
            )
            noise = self.noise_distribution(self.noise_linear(noise))

            hiddens = self.ladder_convs(ladder_input).reshape(
                -1, in_side * 2 * (2 * 4 - 1) ** 2
            )
            hiddens = self.ladder_distribution(self.ladder_linear(hiddens))
        else:
            noise = self.noise_distribution(self.noise_dense(ladder_input))
            hiddens = self.ladder_distribution(self.ladder_dense(ladder_input))

        return hiddens, noise

class LadderPosterior(TypedModel):
    def __init__(self, in_dim, noise_dim, noise_dist):
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = noise_dim

        self.distribution = noise_dist(noise_dim)
        noise_features = noise_dim
        if noise_dist == DiagonalGaussian:
            noise_features *= 2

        self.noise_dense = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.PReLU(),
            nn.Linear(in_dim, noise_features),
        )

    @property
    def type(self):
        return closed.CartesianClosed.ARROW(
            types.tensor_type(torch.float, self._in_dim),
            types.tensor_type(torch.float, self._out_dim)
        )

    @property
    def name(self):
        name = 'q(%s \\mid %s)' % (self.distribution.random_var_name,
                                   '\\mathbb{R}^{%d}' % self._in_dim)
        return '$%s$' % name

    def forward(self, ladder_input):
        return self.distribution(self.noise_dense(ladder_input))

def glimpse_transform(glimpse_code):
    scalings = torch.eye(2).expand(glimpse_code.shape[0], 2, 2).to(glimpse_code)
    scalings = scalings * glimpse_code[:, 0].view(-1, 1, 1)
    return torch.cat((scalings, glimpse_code[:, 1:].unsqueeze(-1)), dim=-1)

def inverse_glimpse(glimpse_code):
    coords = glimpse_code[:, 1:]
    scalars = glimpse_code[:, 0]
    return torch.cat((torch.ones(glimpse_code.shape[0], 1).to(coords), -coords),
                     dim=-1) / scalars.unsqueeze(-1)

class SpatialTransformerWriter(TypedModel):
    def __init__(self, out_dist, canvas_side=28, glimpse_side=7):
        super().__init__()
        self._canvas_side = canvas_side
        self._glimpse_side = glimpse_side
        canvas_name = 'Z^{%d}' % canvas_side ** 2
        self.distribution = out_dist(self._canvas_side ** 2,
                                     observable_name=canvas_name)

    @property
    def type(self):
        canvas_type = types.tensor_type(torch.float, self._canvas_side ** 2)
        glimpse_type = types.tensor_type(torch.float, self._glimpse_side ** 2)
        triple = types.tensor_type(torch.float, 3)

        return closed.CartesianClosed.ARROW(
            closed.CartesianClosed.BASE(Ty(canvas_type, glimpse_type, triple)),
            canvas_type
        )

    @property
    def name(self):
        canvas_name = 'Z^{%d}' % self._canvas_side ** 2
        glimpse_name = 'Z^{%d}' % self._glimpse_side ** 2
        inputs_tuple = ' \\times '.join([canvas_name, glimpse_name,
                                         '\\mathbb{R}^{3}'])
        name = 'p(%s \\mid %s)' % (self.distribution.random_var_name,
                                   inputs_tuple)
        return '$%s$' % name

    def canvas_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._canvas_side,
                           self._canvas_side])

    def glimpse_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._glimpse_side,
                           self._glimpse_side])

    def forward(self, canvas, glimpse_contents, glimpse_params):
        glimpse_transforms = glimpse_transform(glimpse_params)
        grids = F.affine_grid(glimpse_transforms, self.canvas_shape(canvas),
                              align_corners=True)
        glimpse_contents = glimpse_contents.view(*self.glimpse_shape(canvas))
        glimpse = F.grid_sample(glimpse_contents, grids, align_corners=True)
        return self.distribution(canvas + glimpse.view(-1,
                                                       self._canvas_side ** 2))

class SpatialTransformerReader(TypedModel):
    def __init__(self, out_dist, canvas_dist, canvas_side=28, glimpse_side=7):
        super().__init__()
        self._canvas_side = canvas_side
        self._glimpse_side = glimpse_side
        self.glimpse_attention = nn.Sequential(
            nn.Linear(self._canvas_side ** 2, self._canvas_side ** 2),
            nn.LayerNorm(self._canvas_side ** 2), nn.PReLU(),
            nn.Linear(self._canvas_side ** 2, 3 * 2),
        )
        self.coordinates_dist = out_dist(3)
        canvas_name = 'Z^{%d}' % canvas_side ** 2
        self.canvas_dist = canvas_dist(self._canvas_side ** 2,
                                       observable_name=canvas_name)
        glimpse_name = 'Z^{%d}' % glimpse_side ** 2
        self.glimpse_dist = canvas_dist(self._glimpse_side ** 2,
                                        observable_name=glimpse_name)

    @property
    def type(self):
        canvas_type = types.tensor_type(torch.float, self._canvas_side ** 2)
        glimpse_type = types.tensor_type(torch.float, self._glimpse_side ** 2)
        triple = types.tensor_type(torch.float, 3)

        return closed.CartesianClosed.ARROW(
            canvas_type,
            closed.CartesianClosed.BASE(Ty(canvas_type, glimpse_type, triple)),
        )

    @property
    def name(self):
        canvas_name = 'Z^{%d}' % self._canvas_side ** 2
        glimpse_name = 'Z^{%d}' % self._glimpse_side ** 2
        outputs_tuple = ' \\times '.join([canvas_name, glimpse_name,
                                          '\\mathbb{R}^{3}'])
        name = 'q(%s \\mid %s)' % (outputs_tuple, canvas_name)
        return '$%s$' % name

    def canvas_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._canvas_side,
                           self._canvas_side])

    def glimpse_shape(self, imgs):
        return torch.Size([imgs.shape[0], 1, self._glimpse_side,
                           self._glimpse_side])

    def forward(self, images):
        flat_images = images.view(-1, self._canvas_side ** 2)
        coords = self.glimpse_attention(flat_images)
        coords = self.coordinates_dist(coords)
        transforms = glimpse_transform(inverse_glimpse(coords))

        grid = F.affine_grid(transforms, self.glimpse_shape(images),
                             align_corners=True)
        glimpse = F.grid_sample(images.view(*self.canvas_shape(images)), grid,
                                align_corners=True)
        glimpse = self.glimpse_dist(glimpse)

        recon_transforms = glimpse_transform(coords)
        recon_grid = F.affine_grid(recon_transforms, self.canvas_shape(images),
                                   align_corners=True)
        glimpse_recon = F.grid_sample(glimpse.view(*self.glimpse_shape(images)),
                                      recon_grid, align_corners=True)
        glimpse_recon = glimpse_recon.view(-1, self._canvas_side ** 2)
        residual = self.canvas_dist(images - glimpse_recon)

        return residual, glimpse, coords

VAE_MIN_DEPTH = 2

class VAECategoryModel(BaseModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=256):
        super().__init__()
        self._data_dim = data_dim

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        dims = list(util.powers_of(2, hidden_dim, data_dim)) + [49]
        dims.sort()

        generators = []
        for dim_a, dim_b in itertools.combinations(dims, 2):
            lower, higher = sorted([dim_a, dim_b])
            # Construct the decoder
            if higher == self._data_dim:
                decoder = DensityDecoder(lower, higher,
                                         ContinuousBernoulliModel)
            else:
                decoder = DensityDecoder(lower, higher, DiagonalGaussian)
            # Construct the encoder
            encoder = DensityEncoder(higher, lower, DiagonalGaussian)
            in_space, out_space = decoder.type.arrow()
            generator = closed.TypedDaggerBox(decoder.density_name, in_space,
                                              out_space, decoder, encoder,
                                              encoder.density_name)
            generators.append(generator)

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
        for dim in dims:
            noise_space = types.tensor_type(torch.float, 2)
            space = types.tensor_type(torch.float, dim)
            if dim == self._data_dim:
                out_dist = ContinuousBernoulliModel
            else:
                out_dist = DiagonalGaussian
            prior = LadderPrior(2, dim, out_dist)
            posterior = LadderPosterior(dim, 2, DiagonalGaussian)
            generator = closed.TypedDaggerBox(prior.name, noise_space, space,
                                              prior, posterior, posterior.name)
            generators.append(generator)

        # Construct writer/reader pair for spatial attention
        writer = SpatialTransformerWriter(ContinuousBernoulliModel)
        writer_l, writer_r = writer.type.arrow()
        reader = SpatialTransformerReader(DiagonalGaussian,
                                          ContinuousBernoulliModel)
        generator = closed.TypedDaggerBox(writer.name, writer_l, writer_r,
                                          writer, reader, reader.name)
        generators.append(generator)

        global_elements = []
        for dim in {2, 3} | set(dims) - {784}:
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
        self.guide_arrow_distances = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim),
            nn.LayerNorm(guide_hidden_dim), nn.PReLU(),
            nn.Linear(guide_hidden_dim,
                      self._category.arrow_distance_alphas.shape[0] * 2),
            nn.Softplus()
        )

        self._random_variable_names = collections.defaultdict(int)

    @property
    def data_space(self):
        return types.tensor_type(torch.float, self._data_dim)

    @pnn.pyro_method
    def model(self, observations=None):
        if isinstance(observations, dict):
            data = observations['$X^{%d}$' % self._data_dim]
        else:
            data = observations
        if data is None:
            data = torch.zeros(1, self._data_dim)
        data = data.view(data.shape[0], self._data_dim)
        for module in self._category.children():
            if isinstance(module, BaseModel):
                module.set_batching(data)

        morphism = self._category(self.data_space, min_depth=VAE_MIN_DEPTH)
        if observations is not None:
            conditions = {'$X^{%d}$' % self._data_dim: data}
            score_morphism = pyro.condition(morphism, data=conditions)
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
        temperature = pyro.sample('distances_temperature', temperature_gamma)

        data_arrow_distances = self.guide_arrow_distances(data)
        data_arrow_distances = data_arrow_distances.mean(dim=0).view(-1, 2)
        arrow_distances = pyro.sample(
            'arrow_distances',
            dist.Gamma(data_arrow_distances[:, 0],
                       data_arrow_distances[:, 1]).to_event(1)
        )

        morphism = self._category(self.data_space, min_depth=VAE_MIN_DEPTH,
                                  temperature=temperature,
                                  arrow_distances=arrow_distances)
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
