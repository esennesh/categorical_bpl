from collections import OrderedDict
import itertools
import networkx as nx
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import torch
import torch.distributions
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, FirstOrderType, TypedModel
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
    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, self._dim),
            FirstOrderType.TENSORT(torch.float, self._dim)
        )

    def forward(self, inputs, observations=None):
        zs = self.parameterization(inputs).view(-1, 2, self._dim[0])
        normal = dist.Normal(zs[:, 0], F.softplus(zs[:, 1])).to_event(1)
        return pyro.sample(self._latent_name, normal, obs=observations)

class StandardNormal(TypedModel):
    def __init__(self, dim, latent_name=None):
        super().__init__()
        if not latent_name:
            latent_name = 'Z^{%d}' % dim
        self._latent_name = latent_name
        self._dim = dim

    @property
    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TOPT(),
            FirstOrderType.TENSORT(torch.float, torch.Size([self._dim]))
        )

    def forward(self, inputs):
        z_loc = inputs.new_zeros(torch.Size((inputs.shape[0], self._dim)))
        z_scale = inputs.new_ones(torch.Size((inputs.shape[0], self._dim)))
        normal = dist.Normal(z_loc, z_scale).to_event(1)
        return pyro.sample(self._latent_name, normal)

class BernoulliObservation(TypedModel):
    def __init__(self, obs_dim, observable_name=None):
        super().__init__()
        self._obs_dim = torch.Size([obs_dim])
        if not observable_name:
            observable_name = 'X^{%d}' % self._obs_dim[0]
        self._observable_name = observable_name

    @property
    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, self._obs_dim),
            FirstOrderType.TENSORT(torch.float, self._obs_dim)
        )

    def forward(self, inputs, observations=None):
        with name_count():
            xs = torch.sigmoid(inputs.view(-1, self._obs_dim[0]))
            bernoulli = ContinuousBernoulli(probs=xs).to_event(1)
            pyro.sample(self._observable_name, bernoulli, obs=observations)
            return xs

class PathDensityNet(TypedModel):
    def __init__(self, spaces_path, dist_layer=BernoulliObservation):
        super().__init__()
        spaces_path = list(spaces_path)
        self._num_spaces = len(spaces_path)
        self._in_dim = torch.Size([spaces_path[0][0]])
        self._out_dim = torch.Size([spaces_path[-1][-1]])

        layers = []
        for i, (u, v) in enumerate(spaces_path):
            h = u + v // 2
            if i == len(spaces_path) - 1:
                self.add_module('layer_%d' % i, nn.Sequential(
                    nn.Linear(u, h), nn.BatchNorm1d(h), nn.ReLU(),
                    nn.Linear(h, v)
                ))
                self.add_module('distribution', dist_layer(v))
            else:
                self.add_module('layer_%d' % i, nn.Sequential(
                    nn.Linear(u, h), nn.BatchNorm1d(h), nn.ReLU(),
                    nn.Linear(h, v), nn.BatchNorm1d(v), nn.ReLU(),
                ))

    def __len__(self):
        return self._num_spaces

    @property
    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, self._in_dim),
            FirstOrderType.TENSORT(torch.float, self._out_dim)
        )

    def forward(self, inputs, observations=None):
        layers = dict(self.named_children())
        latent = inputs
        for i in range(self._num_spaces):
            latent = layers['layer_%d' % i](latent)
        return self.distribution(latent, observations)

class LayersGraph:
    def __init__(self, spaces, data_dim):
        self._prototype = nx.complete_graph(spaces)
        self._data_space = data_dim

    def latent_spaces(self):
        return {n for n in self._prototype.nodes() if n != self._data_space}

    def likelihoods(self):
        # Use the data space to construct likelihood layers
        for source in self.latent_spaces():
            yield PathDensityNet([(source, self._data_space)],
                                 dist_layer=BernoulliObservation)

    def encoders(self):
        # Use the data space to construct likelihood layers
        for dest in self.latent_spaces():
            yield PathDensityNet([(self._data_space, dest)],
                                 dist_layer=DiagonalGaussian)

    def priors(self):
        for obj in self._prototype:
            yield StandardNormal(obj)

    def latent_maps(self):
        for z1, z2 in itertools.permutations(self.latent_spaces(), 2):
            yield PathDensityNet([(z1, z2)], dist_layer=DiagonalGaussian)

class VAECategoryModel(BaseModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=None):
        super().__init__()
        self._data_dim = data_dim
        self._category = nx.MultiDiGraph()
        self._category.add_node(FirstOrderType.TOPT())
        self._generators = OrderedDict()
        if not guide_hidden_dim:
            guide_hidden_dim = data_dim // 4

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        layers_graph = LayersGraph(util.powers_of(2, hidden_dim, data_dim),
                                   data_dim)
        for k, generator in enumerate(layers_graph.priors()):
            name = 'prior_%d' % k
            self.add_generating_morphism(name, generator)
        for k, generator in enumerate(layers_graph.likelihoods()):
            name = 'likelihood_%d' % k
            self.add_generating_morphism(name, generator)
        for k, generator in enumerate(layers_graph.latent_maps()):
            name = 'latent_map_%d' % k
            self.add_generating_morphism(name, generator)
        for k, generator in enumerate(layers_graph.encoders()):
            name = 'encoder_%d' % k
            self.add_generating_morphism(name, generator)

        latent_dims = torch.ones(len(self._category) - 2)
        latent = lambda s: s != FirstOrderType.TOPT() and s != self.data_space
        latent_subgraph = nx.subgraph_view(self._category, latent)
        for k, space in enumerate(latent_subgraph.nodes()):
            latent_dims[k] = space.tensort()[1][0]
        self.register_buffer('latent_dims', latent_dims)

        self.guide_generator_confidence = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, 2), nn.Softplus(),
        )
        self.guide_generator_weights = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, len(self._generators)), nn.Softplus(),
        )
        self.guide_latent_weights = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, len(self._category) - 2), nn.Softplus(),
        )

    @property
    def data_space(self):
        return FirstOrderType.TENSORT(torch.float, torch.Size([self._data_dim]))

    def add_generating_morphism(self, name, generator):
        assert name not in self._generators
        assert isinstance(generator, TypedModel)

        self._generators[name] = generator
        self.add_module(name, generator)

        l, r = generator.type.arrowt()
        self._category.add_edge(l, r, generator)

    def draw(self):
        diagram = nx.MultiDiGraph(incoming_graph_data=self._category)
        diagram.remove_node(FirstOrderType.TOPT())
        node_labels = {object: str(object) for object in diagram.nodes()}
        nx.draw(diagram, labels=node_labels, pos=nx.spring_layout(diagram))

    def _object_index(self, obj):
        spaces = list(self._category.nodes())
        return spaces.index(obj)

    def _intuitive_distances(self, step_distances):
        transition = step_distances.new_zeros(torch.Size([len(self._category),
                                                   len(self._category)]))
        generators = list(self._generators.values())
        row_indices = []
        column_indices = []
        transition_probs = []
        for src in self._category.nodes():
            i = self._object_index(src)
            out_edges = self._category.out_edges(src, keys=True)
            src_probs = []
            for (_, dest, generator) in out_edges:
                j = self._object_index(dest)
                g = generators.index(generator)
                row_indices.append(i)
                column_indices.append(j)
                src_probs.append(step_distances[g])
            src_probs = F.softmin(torch.stack(src_probs, dim=0), dim=0)
            transition_probs.append(src_probs)

        transition = transition.index_put((torch.LongTensor(row_indices),
                                           torch.LongTensor(column_indices)),
                                          torch.cat(transition_probs, dim=0),
                                          accumulate=True)
        transition_sum = transition.sum(dim=-1, keepdim=True)
        transition = transition / transition_sum
        return util.expm(transition.unsqueeze(0)).squeeze(0)

    def _object_by_dim(self, latent, dims, confidence, infer={}):
        spaces = list(self._category.nodes())
        if latent:
            nonlatent_spaces = [FirstOrderType.TOPT(), FirstOrderType.TENSORT(
                torch.float, torch.Size([self._data_dim])
            )]
            for s in nonlatent_spaces:
                spaces.remove(s)

        dims = F.softmin(dims * confidence, dim=0)
        obj = pyro.sample('category_object', dist.Categorical(probs=dims),
                          infer=infer)

        return spaces[obj.item()]

    def _morphism_by_weight(self, src, dest, weights, confidence, infer={}):
        morphisms = list(self._category[src][dest].keys())
        if len(morphisms) == 1:
            return morphisms[0]
        weights = torch.unbind(weights, dim=0)
        src_dest_weights = []
        for (g, _), w in zip(self._generators.values(), weights):
            if g.type == FirstOrderType.ARROWT(src, dest):
                src_dest_weights += [w]
        src_dest_weights = torch.stack(src_dest_weights, dim=0) * confidence
        morphisms_cat = dist.Categorical(probs=F.softmax(src_dest_weights,
                                                         dim=0))
        k = pyro.sample('morphism_{%s -> %s}' % (src, dest), morphisms_cat,
                        infer=infer)
        return morphisms[k.item()]

    def _morphism_by_distance(self, src, dest, distances, confidence, k=0,
                              infer={}):
        dest_idx = self._object_index(dest)

        morphisms = [(neighbor, m) for (_, neighbor, m) in
                     self._category.out_edges(src, keys=True)]
        if len(morphisms) == 1:
            return morphisms[0]
        j = torch.LongTensor([self._object_index(neighbor) for (neighbor, _) in
                              morphisms]).to(device=distances.device)
        to_dest = distances.index_select(0, j)[:, dest_idx] * confidence
        morphism_cat = dist.Categorical(probs=F.softmax(to_dest, dim=0))
        k = pyro.sample('arrow_%d' % k, morphism_cat, infer=infer)

        return morphisms[k.item()]

    def sample_path_between(self, src, dest, distances, confidence, infer={}):
        location = src
        path = []
        with pyro.markov():
            while location != dest:
                (location, morphism) = self._morphism_by_distance(location,
                                                                  dest,
                                                                  distances,
                                                                  confidence,
                                                                  k=len(path))
                path.append(morphism)

        return path

    def model(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        if data is None:
            data = torch.zeros(1, self._data_dim)
        data = data.view(data.shape[0], self._data_dim)

        alpha = pyro.param('confidence_alpha', data.new_ones(1),
                           constraint=constraints.positive)
        beta = pyro.param('confidence_beta', data.new_ones(1),
                          constraint=constraints.positive)
        confidence_gamma = dist.Gamma(alpha, beta)
        confidence = pyro.sample('generators_confidence',
                                 confidence_gamma.to_event(0))

        weights = []
        for (src, dest, generator) in self._category.edges(keys=True):
            pyro.module('generator_{%s -> %s}' % (src, dest), generator)
            w = pyro.param('generator_weight_{%s -> %s}' % (src, dest),
                           data.new_ones(1), constraint=constraints.positive)
            weights.append(w)
        weights = torch.cat(weights, dim=0)
        distances = self._intuitive_distances(weights)

        origin = self._object_by_dim(True, self.latent_dims, confidence)
        prior = self._morphism_by_weight(FirstOrderType.TOPT(), origin, weights,
                                         confidence)
        path = self.sample_path_between(origin, self.data_space, distances,
                                        confidence)

        with pyro.plate('data', len(data)):
            with pyro.markov():
                with name_count():
                    latent = prior(data)
                    for i, morphism in enumerate(path):
                        if i == len(path) - 1:
                            latent = morphism(latent, observations=data)
                        else:
                            latent = morphism(latent)

        return latent

    def guide(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], self._data_dim)

        pyro.module('guide_generator_confidence',
                    self.guide_generator_confidence)
        pyro.module('guide_generator_weights', self.guide_generator_weights)
        pyro.module('guide_latent_weights', self.guide_latent_weights)

        generators_confidence = self.guide_generator_confidence(data).mean(
            dim=0
        )
        confidence_gamma = dist.Gamma(generators_confidence[0],
                                      generators_confidence[1])
        confidence = pyro.sample('generators_confidence',
                                 confidence_gamma.to_event(0))

        weights = self.guide_generator_weights(data).mean(dim=0)
        distances = self._intuitive_distances(weights)

        latent_dims = self.guide_latent_weights(data).mean(dim=0)
        origin = self._object_by_dim(True, latent_dims, confidence)
        prior = self._morphism_by_weight(FirstOrderType.TOPT(), origin,
                                         weights, confidence)
        path = self.sample_path_between(origin, self.data_space, distances,
                                        confidence)

        encoders = []
        # Cycle through while the location is not the data space, finding
        # a path there via intuitive distance softmin.
        with pyro.markov():
            for arrow in path:
                location = types.unfold_arrow(arrow.type)[0]
                encoder = self._morphism_by_weight(self.data_space, location,
                                                   weights, confidence,
                                                   infer={'is_auxiliary': True})
                encoders.append(encoder)

        with pyro.plate('data', len(data)):
            with name_count():
                for encoder in encoders:
                    latent = encoder(data)

        return latent

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
