from indexed import IndexedOrderedDict
import itertools
import networkx as nx
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import torch
import torch.distributions
import torch.distributions.constraints as constraints
import torch_expm.expm_taylor as expm
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

    @property
    def spaces(self):
        return set(self._prototype.nodes())

    @property
    def latent_spaces(self):
        return {n for n in self._prototype.nodes() if n != self._data_space}

    def likelihoods(self):
        # Use the data space to construct likelihood layers
        for source in self.latent_spaces:
            yield PathDensityNet([(source, self._data_space)],
                                 dist_layer=BernoulliObservation)

    def encoders(self):
        # Use the data space to construct likelihood layers
        for dest in self.latent_spaces:
            yield PathDensityNet([(self._data_space, dest)],
                                 dist_layer=DiagonalGaussian)

    def priors(self):
        for obj in self._prototype:
            yield self.prior(obj)

    def prior(self, obj):
        return StandardNormal(obj)

    def latent_maps(self):
        for z1, z2 in itertools.permutations(self.latent_spaces, 2):
            yield PathDensityNet([(z1, z2)], dist_layer=DiagonalGaussian)

class VAECategoryModel(BaseModel):
    def __init__(self, data_dim=28*28, hidden_dim=64, guide_hidden_dim=None):
        super().__init__()
        self._data_dim = data_dim
        self._category = nx.MultiDiGraph()
        self._generators = IndexedOrderedDict()
        self._spaces = []
        if not guide_hidden_dim:
            guide_hidden_dim = data_dim // 4

        # Build up a bunch of torch.Sizes for the powers of two between
        # hidden_dim and data_dim.
        layers_graph = LayersGraph(util.powers_of(2, hidden_dim, data_dim),
                                   data_dim)
        for dim in layers_graph.spaces:
            space = FirstOrderType.TENSORT(torch.float, torch.Size([dim]))
            self._category.add_node(space, global_elements=tuple())
            self._spaces.append(space)
        for prior in layers_graph.priors():
            self.add_global_element(prior)
        for k, generator in enumerate(layers_graph.likelihoods()):
            name = 'likelihood_%d' % k
            self.add_generating_morphism(name, generator)
        for k, generator in enumerate(layers_graph.latent_maps()):
            name = 'latent_map_%d' % k
            self.add_generating_morphism(name, generator)
        for k, generator in enumerate(layers_graph.encoders()):
            name = 'encoder_%d' % k
            self.add_generating_morphism(name, generator)

        dimensionalities = torch.ones(len(self._category))
        for k, space in enumerate(self._category.nodes()):
            dimensionalities[k] = space.tensort()[1][0]
        self.register_buffer('dimensionalities', dimensionalities)

        self.guide_confidence = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, 2), nn.Softplus(),
        )
        self.guide_prior_weights = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, len(list(layers_graph.priors()))),
            nn.Softplus(),
        )
        self.guide_distances = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, len(self._generators)),
            nn.Softplus(),
        )
        self.guide_dimensionalities = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim), nn.ReLU(),
            nn.Linear(guide_hidden_dim, len(self._category)), nn.Softplus(),
        )

    @property
    def data_space(self):
        return FirstOrderType.TENSORT(torch.float, torch.Size([self._data_dim]))

    def add_global_element(self, element):
        assert isinstance(element, TypedModel)
        assert element.type.arrowt()[0] == FirstOrderType.TOPT()

        obj = element.type.arrowt()[1]
        elements = list(self._category.nodes[obj]['global_elements'])

        self.add_module('global_element_{%s, %d}' % (obj, len(elements)),
                        element)
        elements.append(element)
        self._category.add_node(obj, global_elements=tuple(elements))

    def add_generating_morphism(self, name, generator):
        assert name not in self._generators
        assert isinstance(generator, TypedModel)

        self._generators[generator] = name
        self.add_module(name, generator)

        l, r = generator.type.arrowt()
        self._category.add_edge(l, r, generator)

    def draw(self):
        diagram = nx.MultiDiGraph(incoming_graph_data=self._category)
        node_labels = {object: str(object) for object in diagram.nodes()}
        nx.draw(diagram, labels=node_labels, pos=nx.spring_layout(diagram))

    def _object_index(self, obj):
        return self._spaces.index(obj)

    def _generator_index(self, generator):
        return self._generators.keys().index(generator)

    def _intuitive_distances(self, edge_distances):
        transition = edge_distances.new_zeros(torch.Size([len(self._category),
                                                          len(self._category)]))
        row_indices = []
        column_indices = []
        transition_probs = []
        for src in self._category.nodes():
            i = self._object_index(src)
            out_edges = self._category.out_edges(src, keys=True)
            src_probs = []
            for (_, dest, generator) in out_edges:
                j = self._object_index(dest)
                g = self._generator_index(generator)
                row_indices.append(i)
                column_indices.append(j)
                src_probs.append(edge_distances[g])
            src_probs = F.softmin(torch.stack(src_probs, dim=0), dim=0)
            transition_probs.append(src_probs)

        transition = transition.index_put((torch.LongTensor(row_indices),
                                           torch.LongTensor(column_indices)),
                                          torch.cat(transition_probs, dim=0),
                                          accumulate=True)
        transition_sum = transition.sum(dim=-1, keepdim=True)
        transition = transition / transition_sum
        transition = expm.expm(transition.unsqueeze(0)).squeeze(0)
        transition_sum = transition.sum(dim=-1, keepdim=True)
        transition = transition / transition_sum
        return -torch.log(transition)

    def sample_object(self, dims, confidence, latent=False, infer={}):
        spaces = self._spaces.copy()
        if latent:
            data_idx = spaces.index(self.data_space)
            spaces.remove(self.data_space)
            dims = torch.cat((dims[0:data_idx], dims[data_idx+1:]), dim=0)

        dims = F.softmin(dims * confidence, dim=0)
        obj_idx = pyro.sample('global_object', dist.Categorical(probs=dims),
                              infer=infer)
        return spaces[obj_idx.item()]

    def sample_global_element(self, obj, weights, confidence, latent=False,
                              infer={}):
        elements_cat = dist.Categorical(
            probs=F.softmax(weights[obj] * confidence, dim=0)
        )
        elt_idx = pyro.sample('global_element_{%s}' % obj, elements_cat,
                              infer=infer)
        return self._category.nodes[obj]['global_elements'][elt_idx.item()]

    def edge_navigation_distances(self, object_distances, dest, forward=True):
        dest_idx = self._object_index(dest)

        if forward:
            rows = torch.LongTensor([self._object_index(v) for (_, v)
                                     in self._category.edges()])
            rows = rows.to(device=object_distances.device)
            edge_distances = object_distances[rows, dest_idx]
            assert edge_distances.shape[0] == len(rows)
        else:
            cols = torch.LongTensor([self._object_index(u) for (u, _)
                                     in self._category.edges()])
            cols = cols.to(device=object_distances.device)
            edge_distances = object_distances[dest_idx, cols]
            assert edge_distances.shape[0] == len(cols)

        return edge_distances

    def navigate_morphism(self, src, dest, object_distances, confidence, k=0,
                          infer={}, name='arrow', forward=True):
        edge_distances = self.edge_navigation_distances(object_distances, dest,
                                                        forward=forward)

        if forward:
            morphisms = [(v, g) for (_, v, g) in
                         self._category.out_edges(src, keys=True)]
        else:
            morphisms = [(u, g) for (u, _, g) in
                         self._category.in_edges(src, keys=True)]
        indices = [self._generator_index(g) for (_, g) in morphisms]
        indices = torch.LongTensor(indices).to(device=confidence.device)
        edge_distances = edge_distances[indices]

        morphism_cat = dist.Categorical(probs=F.softmin(
            edge_distances * confidence, dim=0
        ))
        idx = pyro.sample('%s_%d' % (name, k), morphism_cat, infer=infer)

        return morphisms[idx.item()]

    def sample_generator_between(self, edge_distances, confidence, src=None,
                                 dest=None, infer={}, name='generator',
                                 exclude=[]):
        assert src or dest
        if src and dest:
            generators = list(self._category[src][dest].keys())
        elif src:
            generators = [(v, g) for (_, v, g) in
                          self._category.out_edges(src, keys=True)
                          if v not in exclude]
        elif dest:
            generators = [(u, g) for (u, _, g) in
                          self._category.in_edges(dest, keys=True)
                          if u not in exclude]
        if len(generators) == 1:
            return generators[0]

        edge_distances = torch.unbind(edge_distances, dim=0)
        between_distances = []

        for (_, generator) in generators:
            g = self._generator_index(generator)
            between_distances.append(edge_distances[g])
        between_distances = torch.stack(between_distances, dim=0)

        generators_cat = dist.Categorical(
            probs=F.softmin(between_distances * confidence, dim=0)
        )
        g_idx = pyro.sample('%s_{%s -> %s}' % (name, src, dest), generators_cat,
                            infer=infer)
        return generators[g_idx.item()]

    def sample_path_to(self, dest, edge_distances, confidence, infer={}):
        bernoulli = dist.Bernoulli(probs=(-(1. / confidence)).exp())

        location = dest
        path = []
        loop = torch.tensor([True]).to(dtype=torch.bool,
                                       device=confidence.device)
        with pyro.markov():
            while loop.item():
                (location, morphism) = self.sample_generator_between(
                    edge_distances, confidence, dest=location, infer=infer,
                    name='generator_%d' % -len(path), exclude=[dest]
                )
                path.append(morphism)
                loop = pyro.sample('path_continues_%d' % len(path), bernoulli)
                loop = loop.to(dtype=torch.bool)
        return list(reversed(path))

    def sample_path_between(self, src, dest, distances, confidence, infer={}):
        location = src
        path = []
        with pyro.markov():
            while location != dest:
                (location, morphism) = self.navigate_morphism(location, dest,
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

        distances = []
        for (src, dest, generator) in self._category.edges(keys=True):
            pyro.module('generator_{%s -> %s}' % (src, dest), generator)
            d = pyro.param('generator_weight_{%s -> %s}' % (src, dest),
                           data.new_ones(1), constraint=constraints.positive)
            distances.append(d)
        distances = self._intuitive_distances(torch.cat(distances, dim=0))

        prior_weights = {}
        for obj in self._category.nodes:
            prior_weights[obj] = []
            global_elements = self._category.nodes[obj]['global_elements']
            for k, element in enumerate(global_elements):
                pyro.module('global_element_%s_%d' % (obj, k), element)
                weight = pyro.param('global_element_weight_%s_%s' % (obj, k),
                                    data.new_ones(1),
                                    constraint=constraints.positive)
                prior_weights[obj].append(weight)
            prior_weights[obj] = torch.stack(prior_weights[obj], dim=0)

        alpha = pyro.param('confidence_alpha', data.new_ones(1),
                           constraint=constraints.positive)
        beta = pyro.param('confidence_beta', data.new_ones(1),
                          constraint=constraints.positive)
        confidence_gamma = dist.Gamma(alpha, beta)
        confidence = pyro.sample('generators_confidence',
                                 confidence_gamma.to_event(0))

        origin = self.sample_object(self.dimensionalities, confidence,
                                    latent=True)
        prior = self.sample_global_element(origin, prior_weights, confidence,
                                           latent=True)
        path = self.sample_path_between(origin, self.data_space, distances,
                                        confidence)

        with pyro.plate('data', len(data)):
            with pyro.markov():
                with name_count():
                    latent = prior(data)
                    for i, generator in enumerate(path):
                        if i == len(path) - 1:
                            latent = generator(latent, observations=data)
                        else:
                            latent = generator(latent)

        return latent

    def guide(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], self._data_dim)

        pyro.module('guide_confidence', self.guide_confidence)
        pyro.module('guide_distances', self.guide_distances)
        pyro.module('guide_prior_weights', self.guide_prior_weights)
        pyro.module('guide_dimensionalities', self.guide_dimensionalities)

        generators_confidence = self.guide_confidence(data).mean(
            dim=0
        )
        confidence_gamma = dist.Gamma(generators_confidence[0],
                                      generators_confidence[1])
        confidence = pyro.sample('generators_confidence',
                                 confidence_gamma.to_event(0))

        weights = self.guide_prior_weights(data).mean(dim=0)
        prior_weights = {}
        n_prior_weights = 0
        for obj in self._category.nodes:
            prior_weights[obj] = []
            global_elements = self._category.nodes[obj]['global_elements']
            for element in global_elements:
                prior_weights[obj].append(weights[n_prior_weights])
                n_prior_weights += 1
            prior_weights[obj] = torch.stack(prior_weights[obj], dim=0)

        edge_distances = self.guide_distances(data).mean(dim=0)
        distances = self._intuitive_distances(edge_distances)

        dimensionalities = self.guide_dimensionalities(data).mean(dim=0)
        origin = self.sample_object(dimensionalities, confidence, latent=True)
        prior = self.sample_global_element(origin, prior_weights, confidence,
                                           latent=True)
        path = self.sample_path_between(origin, self.data_space, distances,
                                        confidence)

        encoders = []
        # Walk through the sampled path, obtaining an independent encoder from
        # the data space for each step.
        for k, arrow in enumerate(path):
            location = types.unfold_arrow(arrow.type)[0]
            encoder = self.sample_generator_between(
                edge_distances, confidence, src=self.data_space, dest=location,
                infer={'is_auxiliary': True}, name='encoder'
            )
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
