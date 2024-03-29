import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler, SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    @property
    def batch_length(self):
        return self.batch_size

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

class BaseTargetBatchDataLoader(DataLoader):
    """
    Base class for data loaders that batch over classes
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split,
                 num_workers, targets, collate_fn=default_collate,
                 drop_train_last=True, drop_valid_last=True, evaluation=None):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split,
                                                               targets, batch_size,
                                                               self.n_samples,
                                                               drop_train_last,
                                                               drop_valid_last)

        self._batch_length = batch_size
        self.init_kwargs = {
            'dataset': dataset,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(batch_sampler=self.sampler, **self.init_kwargs)
        if evaluation:
            self.eval_dataset, eval_targets = evaluation
            valid_idx = np.arange(len(self.eval_dataset))

            self.valid_sampler = TargetBatchRandomSampler(valid_idx, eval_targets[valid_idx],
                                                          batch_size, drop_valid_last)
        else:
            self.eval_dataset = None

    @property
    def batch_length(self):
        return self._batch_length

    def _split_sampler(self, split, targets, batch_size, n_samples,
                       drop_train_last=True, drop_valid_last=True):
        if split == 0.0:
            return None, None

        idx_full = np.arange(n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = TargetBatchRandomSampler(train_idx, targets[train_idx],
                                                 batch_size, drop_train_last)
        valid_sampler = TargetBatchRandomSampler(valid_idx, targets[valid_idx],
                                                 batch_size, drop_valid_last)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            init_kwargs = dict(self.init_kwargs)
            if self.eval_dataset:
                init_kwargs['dataset'] = self.eval_dataset
            return DataLoader(batch_sampler=self.valid_sampler, **init_kwargs)

class TargetBatchRandomSampler(Sampler):
    def __init__(self, indices, targets, batch_size, drop_last=False):
        if not isinstance(batch_size, int) or\
           isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.indices = indices
        self.targets = targets
        self._targets = {t.item() for t in targets}
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = {t: [] for t in self._targets}
        for idx in torch.randperm(len(self.indices)):
            target = self.targets[idx].item()
            batches[target].append(self.indices[idx])
            if len(batches[target]) == self.batch_size:
                yield batches[target]
                batches[target] = []
        remainders = list(batches.values())
        for idx in torch.randperm(len(remainders)):
            if len(remainders[idx]) > 0 and not self.drop_last:
                yield remainders[idx]

    def __len__(self):
        return len([batch for batch in self])
