import numpy as np
import torch
from torch._six import int_classes as _int_classes
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
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split,
                                                               dataset.targets,
                                                               batch_size)
        self._batch_length = batch_size
        self.init_kwargs = {
            'dataset': dataset,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(batch_sampler=self.sampler, **self.init_kwargs)

    @property
    def batch_length(self):
        return self._batch_length

    def _split_sampler(self, split, targets, batch_size):
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

        train_sampler = TargetBatchRandomSampler(train_idx, targets[train_idx],
                                                 batch_size)
        valid_sampler = TargetBatchRandomSampler(valid_idx, targets[valid_idx],
                                                 batch_size)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(batch_sampler=self.valid_sampler, **self.init_kwargs)

class TargetBatchRandomSampler(Sampler):
    def __init__(self, indices, targets, batch_size, drop_last=False):
        if not isinstance(batch_size, _int_classes) or\
           isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last

        target_set = {t.item() for t in targets}
        target_indices = [indices[targets == t] for t in target_set]
        self._batches = []
        for t, t_indices in zip(target_set, target_indices):
            i = 0
            while len(t_indices) - i * self.batch_size >= self.batch_size:
                k = i * self.batch_size
                self._batches.append(t_indices[k:k+self.batch_size])
                i += 1
            if len(t_indices) - i * self.batch_size > 0 and not self.drop_last:
                self._batches.append(t_indices[i * self.batch_size:])

    def __iter__(self):
        batch = []
        for b in torch.randperm(len(self._batches)):
            yield np.random.permutation(self._batches[b])

    def __len__(self):
        return len(self._batches)
