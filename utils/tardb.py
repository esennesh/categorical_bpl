"""Utilities for topographic factor analysis"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'sennesh.e@husky.neu.edu',
             'khan.zu@husky.neu.edu')
import torch
import torch.utils.data
import webdataset as wds
from ordered_set import OrderedSet
from . import utils

def unique_properties(key_func, data):
    results = []
    result_set = set()

    for rec in data:
        prop = key_func(rec)
        if prop.__hash__:
            prop_ = prop
        else:
            prop_ = str(prop)
        if prop_ not in result_set:
            results.append(prop)
            result_set.add(prop_)

    return results

def _collation_fn(samples):
    result = {'__key__': [], 'activations': [], 't': [], 'block': []}
    for sample in samples:
        for k, v in sample.items():
            result[k].append(v)

    result['activations'] = torch.stack(result['activations'], dim=0)
    result['t'] = torch.tensor(result['t'], dtype=torch.long)
    result['block'] = torch.tensor(result['block'], dtype=torch.long)

    return result

class FmriTarDataset:
    def __init__(self, path, verbose_caching=False):
        self._verbose_caching = verbose_caching

        self._metadata = torch.load(path + '.meta')
        self._path = path
        self._num_times = self._metadata['num_times']

        self._dataset = wds.WebDataset(path, length=self._num_times)
        self._dataset = self._dataset.decode().rename(
            activations='pth', t='time.index', block='block.id',
            __key__='__key__'
        )
        self._dataset = self._dataset.map_dict(
            activations=lambda acts: acts.to_dense()
        )
        self.voxel_locations = self._metadata['voxel_locations']

        self._blocks = {}
        for block in self._metadata['blocks']:
            self._blocks[block['block']] = {
                'id': block['block'],
                'indices': [],
                'individual_differences': block['individual_differences'],
                'run': block['run'],
                'subject': block['subject'],
                'task': block['task'],
                'template': block['template'],
                'times': []
            }
        for k, tr in enumerate(self._dataset):
            self.blocks[tr['block']]['indices'].append(k)
            self.blocks[tr['block']]['times'].append(tr['t'])

    @property
    def blocks(self):
        return self._blocks

    def data(self, batch_size=None, selector=None):
        result = self._dataset
        result_len = len(result)
        if selector:
            selected_len = 0
            for tr in result:
                if selector(tr):
                    selected_len += 1

            result_len = selected_len
            result = result.select(selector)
        if batch_size:
            result = result.batched(batch_size, _collation_fn)
        db_path = self._path + '_' + str(hash(selector)) + '.db'
        return result.compose(wds.DBCache, db_path, result_len,
                              verbose=self._verbose_caching)

    def __getitem__(self, b):
        block = self.blocks[b]
        data = self._dataset.slice(block['indices'][0],
                                   block['indices'][-1] + 1)
        return _collation_fn(data)

    def inference_filter(self, training=True, held_out_subjects=set(),
                         held_out_tasks=set()):
        subjects = OrderedSet([s for s in self.subjects()
                               if s not in held_out_subjects])
        tasks = OrderedSet([t for t in self.tasks() if t not in held_out_tasks])
        diagonals = frozenset(utils.striping_diagonal_indices(len(subjects),
                                                              len(tasks)))
        def result(b):
            if 'block' in b:
                subject = self._blocks[b['block']]['subject']
                task = self._blocks[b['block']]['task']
            else:
                subject, task = b['subject'], b['task']

            subject_index = subjects.index(subject)
            task_index = tasks.index(task)
            return ((subject_index, task_index) in diagonals) == (not training)

        return result

    def _mean_block(self):
        num_times = max(row['t'] for row in self._dataset) + 1
        mean = torch.zeros(num_times, self.voxel_locations.shape[0])
        for tr in self._dataset:
            mean[tr['t']] += tr['activations']
        return mean / len(self.blocks)

    def mean_block(self, save=False):
        if 'mean_block' in self._metadata:
            return self._metadata['mean_block']

        mean = self._mean_block()
        if save:
            self._metadata['mean_block'] = mean
            torch.save(self._metadata, self._path + '.meta')

        return mean

    def _normalize_activations(self):
        subject_runs = self.subject_runs()
        run_activations = {(subject, run): [] for subject, run in subject_runs}
        for tr in self._dataset:
            subject = self.blocks[tr['block']]['subject']
            run = self.blocks[tr['block']]['run']
            run_activations[(subject, run)].append(tr['activations'])
        for sr, acts in run_activations.items():
            run_activations[sr] = torch.stack(acts, dim=0).flatten()

        normalizers = []
        sufficient_stats = []
        for block in self.blocks.values():
            activations = run_activations[(block['subject'], block['run'])]
            normalizers.append(torch.abs(activations).max())
            sufficient_stats.append((torch.mean(activations, dim=0),
                                     torch.std(activations, dim=0)))

        return normalizers, sufficient_stats

    def normalize_activations(self, save=False):
        if 'normalizer_stats' in self._metadata:
            return self._metadata['normalizer_stats']

        normalizers, sufficient_stats = self._normalize_activations()
        if save:
            self._metadata['normalizer_stats'] = (normalizers, sufficient_stats)
            torch.save(self._metadata, self._path + '.meta')

        return normalizers, sufficient_stats

    def runs(self):
        return unique_properties(lambda b: b['run'], self.blocks.values())

    def subjects(self):
        return unique_properties(lambda b: b['subject'], self.blocks.values())

    def subject_runs(self):
        return unique_properties(lambda b: (b['subject'], b['run']),
                                 self.blocks.values())

    def tasks(self):
        return unique_properties(lambda b: b['task'], self.blocks.values())

    def templates(self):
        return unique_properties(lambda b: b['template'], self.blocks.values())
