import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader, BaseTargetBatchDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistTargetBatchDataLoader(BaseTargetBatchDataLoader):
    """
    MNIST data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, self.dataset.targets, num_workers)

rescaling = lambda x : (x - .5) * 2.
flip = lambda x : - x
resizing = lambda x: x.resize((28,28))
omni_transforms = transforms.Compose([resizing, transforms.ToTensor(), rescaling, flip])

class OmniglotTargetBatchDataLoader(BaseTargetBatchDataLoader):
    """
    Omniglot data loading batched by label using BaseTargetBatchDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, background=True):
        self.data_dir = data_dir
        self.dataset = datasets.Omniglot(self.data_dir, background=background, download=True, transform=omni_transforms)
        self.targets = np.array(list(map(lambda x: x[1], self.dataset._flat_character_images)))
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, self.targets, drop_train_last=False,
                         drop_valid_last=False)
