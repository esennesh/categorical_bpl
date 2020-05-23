from indexed import IndexedOrderedDict
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pytorch_expm.expm_taylor as expm
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
    def latent_name(self):
        return self._latent_name

    @property
    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, self._dim),
            FirstOrderType.TENSORT(torch.float, self._dim)
        )

    def forward(self, inputs, observations=None, sample=True):
        zs = self.parameterization(inputs).view(-1, 2, self._dim[0])
        mean, std_dev = zs[:, 0], F.softplus(zs[:, 1].exp())
        if sample:
            normal = dist.Normal(mean, std_dev).to_event(1)
            return pyro.sample(self._latent_name, normal, obs=observations)
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

    def forward(self, inputs, observations=None, sample=True):
        with name_count():
            xs = torch.sigmoid(inputs.view(-1, self._obs_dim[0]))
            if sample:
                bernoulli = ContinuousBernoulli(probs=xs).to_event(1)
                pyro.sample(self._observable_name, bernoulli, obs=observations)
            return xs

class PathDensityNet(TypedModel):
    def __init__(self, in_dim, out_dim, dist_layer=BernoulliObservation):
        super().__init__()
        self._in_space = FirstOrderType.TENSORT(torch.float,
                                                torch.Size([in_dim]))
        self._out_space = FirstOrderType.TENSORT(torch.float,
                                                 torch.Size([out_dim]))

        hidden_dim = in_dim + out_dim // 2
        self.add_module('residual_layer', nn.Sequential(
            nn.BatchNorm1d(in_dim), nn.PReLU(), nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        ))
        self.add_module('identity_layer', nn.Linear(in_dim, out_dim))
        self.add_module('distribution', dist_layer(out_dim))

    @property
    def type(self):
        return FirstOrderType.ARROWT(self._in_space, self._out_space)

    def forward(self, inputs, observations=None, sample=True):
        hidden = self.residual_layer(inputs) + self.identity_layer(inputs)
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
                                 dist_layer=BernoulliObservation)

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
        self._category = nx.MultiDiGraph()
        self._generators = IndexedOrderedDict()
        self._spaces = []
        if not guide_hidden_dim:
            guide_hidden_dim = data_dim // 16

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

        self.guide_embedding = nn.Sequential(
            nn.Linear(data_dim, guide_hidden_dim),
            nn.BatchNorm1d(guide_hidden_dim), nn.PReLU(),
        )
        self.guide_confidences = nn.Sequential(
            nn.Linear(guide_hidden_dim, 1 * 2), nn.Softplus(),
        )

        max_generators = 0
        for obj in self._category:
            _, num_edges, num_priors = self._object_generators(obj, False)
            num_generators = num_edges + num_priors
            if num_generators > max_generators:
                max_generators = num_generators
        self.guide_navigator = nn.GRUCell(len(self._category), guide_hidden_dim)
        self.guide_navigation_decoder = nn.Sequential(
            nn.Linear(guide_hidden_dim, max_generators), nn.Softplus()
        )

        self.register_buffer('edge_distances',
                             torch.zeros(len(self._category.edges)))

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

    def global_element_weights(self, weights=None):
        prior_weights = {}
        if weights is not None:
            n_prior_weights = 0
        for obj in self._category.nodes:
            prior_weights[obj] = []
            global_elements = self._category.nodes[obj]['global_elements']
            for k, element in enumerate(global_elements):
                if weights is not None:
                    weight = weights[n_prior_weights]
                    n_prior_weights += 1
                else:
                    pyro.module('global_element_%s_%d' % (obj, k), element)
                    weight_name = 'global_element_weight_%s_%s' % (obj, k)
                    weight = pyro.param(weight_name,
                                        self.edge_distances.new_ones(()),
                                        constraint=constraints.positive)
                prior_weights[obj].append(weight)
            prior_weights[obj] = torch.stack(prior_weights[obj], dim=0)

        return prior_weights

    def add_generating_morphism(self, name, generator):
        assert name not in self._generators
        assert isinstance(generator, TypedModel)

        self._generators[generator] = name
        self.add_module(name, generator)

        l, r = generator.type.arrowt()
        self._category.add_edge(l, r, generator)

    def draw(self, path=None, filename=None):
        if path:
            path = set(path)
            def filter_node(node):
                out_keys = {key for (u, v, key) in
                            self._category.out_edges(node, keys=True)}
                in_keys = {key for (u, v, key) in
                           self._category.in_edges(node, keys=True)}
                return len((out_keys | in_keys) & path) > 0
            def filter_edge(u, v, g):
                return g in path
            diagram = nx.subgraph_view(self._category, filter_node=filter_node,
                                       filter_edge=filter_edge)
        else:
            diagram = self._category
        node_labels = {object: str(object) for object in diagram.nodes()}
        nx.draw(diagram, labels=node_labels, pos=nx.spring_layout(diagram))

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def _object_index(self, obj):
        return self._spaces.index(obj)

    def _object_index_onehot(self, obj):
        eye = torch.eye(len(self._category)).to(self.edge_distances)
        return eye[self._object_index(obj)]

    def _generator_index(self, generator):
        return self._generators.keys().index(generator)

    def _generator_distances(self, edge_distances, generators):
        generator_indices = [self._generator_index(g) for (_, g) in generators]
        return edge_distances[generator_indices]

    def get_object_distances(self):
        transition = self.edge_distances.new_zeros((len(self._category),
                                                    len(self._category)))
        row_indices = []
        column_indices = []
        transition_probs = []
        for src in self._category.nodes():
            i = self._object_index(src)
            generators, num_edges, _ = self._object_generators(src)
            generators = generators[:num_edges]
            for (dest, _) in generators:
                j = self._object_index(dest)
                row_indices.append(i)
                column_indices.append(j)
            transition_probs.append(F.softmin(
                self._generator_distances(self.edge_distances, generators),
                dim=0
            ))

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

    def sample_object(self, dims, confidence, exclude=[], k=None, infer={}):
        spaces = self._spaces.copy()
        spaces_indices = list(range(len(spaces)))
        for space in exclude:
            spaces_indices.remove(spaces.index(space))
            spaces.remove(space)
        dims = F.softmin(dims[spaces_indices] * confidence, dim=0)
        name = 'global_object' if k is None else 'global_object_%d' % k
        obj_idx = pyro.sample(name, dist.Categorical(probs=dims), infer=infer)
        return spaces[obj_idx.item()]

    def sample_global_element(self, obj, weights, confidence, infer={}):
        elements_cat = dist.Categorical(
            probs=F.softmax(weights[obj] * confidence, dim=0)
        )
        elt_idx = pyro.sample('global_element_{%s}' % obj, elements_cat,
                              infer=infer)
        return self._category.nodes[obj]['global_elements'][elt_idx.item()]

    def navigate_morphism(self, src, dest, object_distances, confidence, k=0,
                          infer={}, forward=True, edge_costs=None):
        if edge_costs is None:
            edge_costs = self.edge_distances

        dest_idx = self._object_index(dest)
        if forward:
            object_distances = object_distances[:, dest_idx]
        else:
            object_distances = object_distances[dest_idx, :]
        loc = self.sample_object(object_distances, confidence, exclude=[src],
                                 k=k, infer=infer)

        morphism = self.sample_generator_between(edge_costs, confidence,
                                                 src=src, dest=loc, infer=infer,
                                                 name='generator_%d' % k)
        return loc, morphism

    def _object_generators(self, obj, forward=True, excluded=[]):
        edges = self._category.out_edges if forward else self._category.in_edges
        dir_idx = 1 if forward else 0

        generators = [(generator[dir_idx], generator[2]) for generator
                      in edges(obj, keys=True)
                      if generator[dir_idx] not in excluded]
        num_edges = len(generators)
        num_priors = 0
        if not (forward or FirstOrderType.TOPT() in excluded):
            priors = [(FirstOrderType.TOPT(), prior) for prior
                      in self._category.nodes[obj]['global_elements']]
            num_priors += len(priors)
            generators += priors

        return generators, num_edges, num_priors

    def sample_generator_to(self, prior_weights, confidence, dest, infer={},
                            name='generator', penalty=0, excluded_srcs=[],
                            embedding=None):
        generators, num_edges, _ = self._object_generators(dest, False,
                                                           excluded_srcs)

        if embedding is not None:
            gen_distances = self.guide_navigation_decoder(embedding)
            _, num_edges, num_priors = self._object_generators(dest,
                                                               False,
                                                               excluded_srcs)
            generator_distances = gen_distances[:num_edges]
            prior_weights = {
                dest: -gen_distances[num_edges:num_edges+num_priors]
            }
        else:
            generator_distances = self._generator_distances(
                self.edge_distances, generators[:num_edges]
            )
        generator_distances = generator_distances + penalty
        if FirstOrderType.TOPT() not in excluded_srcs:
            generator_distances = torch.cat((generator_distances,
                                             -prior_weights[dest]),
                                            dim=0)
        generators_cat = dist.Categorical(
            probs=F.softmin(generator_distances * confidence, dim=0)
        )
        g_idx = pyro.sample('%s_{ -> %s}' % (name, dest), generators_cat,
                            infer=infer)
        return generators[g_idx.item()]

    def sample_generator_between(self, edge_costs, confidence, src=None,
                                 dest=None, infer={}, name='generator',
                                 exclude=[]):
        assert src or dest
        if src and dest:
            generators = list(self._category[src][dest])
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

        between_distances = self._generator_distances(edge_costs, generators)
        generators_cat = dist.Categorical(
            probs=F.softmin(between_distances * confidence, dim=0)
        )
        g_idx = pyro.sample('%s_{%s -> %s}' % (name, src, dest), generators_cat,
                            infer=infer)
        return generators[g_idx.item()]

    def sample_path_to(self, dest, prior_weights, confidence, embedding=None,
                       infer={}):
        location = dest
        path = []
        with pyro.markov():
            exclude = [FirstOrderType.TOPT(), dest]
            while location != FirstOrderType.TOPT():
                (location, morphism) = self.sample_generator_to(
                    prior_weights, confidence, location,
                    infer=infer, name='generator_%d' % -len(path),
                    penalty=len(path), excluded_srcs=exclude,
                    embedding=embedding
                )
                path.append(morphism)
                exclude = [dest]

                if embedding is not None and location != FirstOrderType.TOPT():
                    embedding = embedding.unsqueeze(0)
                    loc = self._object_index_onehot(location).unsqueeze(0)
                    embedding = self.guide_navigator(loc, embedding).squeeze(0)

        return list(reversed(path))

    def sample_path_between(self, src, dest, distances, confidence, infer={},
                            embedding=None):
        if embedding is not None:
            embedding = embedding.unsqueeze(0)
        location = src
        path = []
        with pyro.markov():
            while location != dest:
                (location, morphism) = self.navigate_morphism(location, dest,
                                                              distances,
                                                              confidence,
                                                              k=len(path),
                                                              infer=infer)
                path.append(morphism)

        return path

    def get_edge_distances(self):
        distances = []

        for (src, dest, generator) in self._category.edges(keys=True):
            pyro.module('generator_{%s -> %s}' % (src, dest), generator)
            d = pyro.param('generator_distance_{%s -> %s}' % (src, dest),
                           self.edge_distances.new_ones(1),
                           constraint=constraints.positive)
            distances.append(d)

        self.edge_distances = torch.cat(distances, dim=0)
        return self.edge_distances

    def model(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        if data is None:
            data = torch.zeros(1, self._data_dim)
        data = data.view(data.shape[0], self._data_dim)

        self.get_edge_distances()

        prior_weights = self.global_element_weights()

        alpha = pyro.param('distances_alpha', data.new_ones(1),
                           constraint=constraints.positive)
        beta = pyro.param('distances_beta', data.new_ones(1),
                          constraint=constraints.positive)
        confidence = pyro.sample('distances_confidence',
                                 dist.Gamma(alpha, beta).to_event(0))
        path = self.sample_path_to(self.data_space, prior_weights, confidence)
        prior = path[0]
        path = path[1:]

        rvs = []
        with pyro.plate('data', len(data)):
            with pyro.markov():
                with name_count():
                    latent = prior(data)
                    rvs.append(latent)
                    for i, generator in enumerate(path):
                        if i == len(path) - 1:
                            rvs.append(generator(latent, observations=data))
                        else:
                            latent = generator(latent)
                            rvs.append(latent)

        return path, rvs[:-1], rvs[-1]

    def guide(self, observations=None):
        if isinstance(observations, dict):
            data = observations['X^{%d}' % self._data_dim]
        else:
            data = observations
        data = data.view(data.shape[0], self._data_dim)

        pyro.module('guide_embedding', self.guide_embedding)
        pyro.module('guide_confidences', self.guide_confidences)
        pyro.module('guide_navigator', self.guide_navigator)
        pyro.module('guide_navigation_decoder', self.guide_navigation_decoder)

        embedding = self.guide_embedding(data).mean(dim=0)

        confidences = self.guide_confidences(embedding).view(1, 2)

        prior_weights = self.global_element_weights()

        self.get_edge_distances()

        confidence_gamma = dist.Gamma(confidences[0, 0],
                                      confidences[0, 1]).to_event(0)
        confidence = pyro.sample('distances_confidence', confidence_gamma)
        path = self.sample_path_to(self.data_space, prior_weights, confidence,
                                   embedding)[1:]

        latents = []
        # Walk through the sampled path, obtaining an independent encoder from
        # the data space for each step, and fusing its prediction with that from
        # the generative model.
        with pyro.plate('data', len(data)):
            with pyro.markov():
                with name_count():
                    for k, arrow in enumerate(path):
                        location = types.unfold_arrow(arrow.type)[0]
                        encoder = self.sample_generator_between(
                            self.edge_distances, confidence,
                            src=self.data_space, dest=location,
                            infer={'is_auxiliary': True}, name='encoder'
                        )

                        if latents:
                            encoding = encoder(data, sample=False)
                            prediction = path[k-1](latents[-1], sample=False)

                            precision = encoding[1] + prediction[1]
                            mean = (encoding[0] * encoding[1] +\
                                    prediction[0] * prediction[1]) / precision
                            std_dev = 1. / precision.sqrt()
                            normal = dist.Normal(mean, std_dev).to_event(1)
                            latent_name = path[k-1].distribution.latent_name
                            latent = pyro.sample(latent_name, normal)
                            latents.append(latent)
                        else:
                            latents.append(encoder(data))

        return path, latents

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
