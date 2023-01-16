import numpy as np
import os.path as path
import torch.utils.data
from torchvision import datasets, transforms
from base import BaseDataLoader, BaseTargetBatchDataLoader
from utils import mol_utils

class IndexedMnist(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return (img, target, index)

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, indexed=False):
        trsfm = transforms.ToTensor()
        self.data_dir = data_dir
        dataset = IndexedMnist if indexed else datasets.MNIST
        self.dataset = dataset(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistTargetBatchDataLoader(BaseTargetBatchDataLoader):
    """
    MNIST data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, indexed=False):
        trsfm = transforms.ToTensor()
        self.data_dir = data_dir
        dataset = IndexedMnist if indexed else datasets.MNIST
        self.dataset = dataset(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, self.dataset.targets)

flip = lambda x : 1 - x
resizing = lambda x: x.resize((28,28))
omni_transforms = transforms.Compose([resizing, transforms.ToTensor(), flip])

class OmniglotTargetTransform:
    def __init__(self, data_dir, background=True):
        target_folder = 'images_background' if background else 'images_evaluation'
        target_folder = path.join(data_dir + '/omniglot-py', target_folder)
        self._character_alphabets = []
        for a, alphabet in enumerate(datasets.utils.list_dir(target_folder)):
            alphabet_dir = path.join(target_folder, alphabet)
            for character in datasets.utils.list_dir(alphabet_dir):
                self._character_alphabets.append(a)

    def __call__(self, flat_character_class):
        return self._character_alphabets[flat_character_class]

class OmniglotDataLoader(BaseDataLoader):
    """
    Omniglot data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, background=True):
        self.data_dir = data_dir
        self.dataset = datasets.Omniglot(self.data_dir, background=background, download=True, transform=omni_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class OmniglotTargetBatchDataLoader(BaseTargetBatchDataLoader):
    """
    Omniglot data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        dataset = datasets.Omniglot(self.data_dir, download=True, background=True)
        eval_dataset = datasets.Omniglot(self.data_dir, download=True, background=False)
        target_transform = OmniglotTargetTransform(self.data_dir, background=True)
        eval_target_transform = OmniglotTargetTransform(self.data_dir, background=False)

        self.dataset = datasets.Omniglot(self.data_dir, background=True, download=True, transform=omni_transforms, target_transform=target_transform)
        self.eval_dataset = datasets.Omniglot(self.data_dir, background=False, download=True, transform=omni_transforms, target_transform=eval_target_transform)
        self.targets = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        eval_targets = np.array([self.eval_dataset[i][1] for i in range(len(self.eval_dataset))])
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, self.targets, drop_train_last=False,
                         drop_valid_last=False, evaluation=(self.eval_dataset, eval_targets))

class FashionMnistDataLoader(BaseDataLoader):
    """
    Fashion-MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True):
        trsfm = transforms.ToTensor()
        self.data_dir = data_dir
        self.dataset = datasets.FashionMNIST(self.data_dir, train=training,
                                             download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)

class FashionMnistTargetBatchDataLoader(BaseTargetBatchDataLoader):
    """
    Fashion-MNIST data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.ToTensor()
        self.data_dir = data_dir
        self.dataset = datasets.FashionMNIST(self.data_dir, train=training,
                                             download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, self.dataset.targets)

class ZincMolecularDataLoader(BaseDataLoader):
    """
    ZINC data loading using BaseDataLoader
    """
    def __init__(self, csv, batch_size, max_length=120, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        dataset = mol_utils.Zinc15Dataset(csv, max_len=max_length)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
