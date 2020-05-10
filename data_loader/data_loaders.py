import numpy as np
import os.path as path
from torchvision import datasets, transforms
from base import BaseDataLoader, BaseTargetBatchDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.ToTensor()
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistTargetBatchDataLoader(BaseTargetBatchDataLoader):
    """
    MNIST data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.ToTensor()
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, self.dataset.targets)

rescaling = lambda x : (x - .5) * 2.
flip = lambda x : - x
resizing = lambda x: x.resize((28,28))
omni_transforms = transforms.Compose([resizing, transforms.ToTensor(), rescaling, flip])

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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, background=True):
        self.data_dir = data_dir
        target_transform = OmniglotTargetTransform(self.data_dir, background=background)
        self.dataset = datasets.Omniglot(self.data_dir, background=background, download=True, transform=omni_transforms, target_transform=target_transform)
        self.targets = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, self.targets, drop_train_last=False,
                         drop_valid_last=False)

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
