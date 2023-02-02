import pyro
import torch
import numpy as np
from abc import abstractmethod, abstractproperty


class BaseModel(pyro.nn.PyroModule):
    """
    Base class for all models
    """
    def __init__(self):
        super().__init__()
        self._batch = None
        self._strict_load = True

    def set_batching(self, batch):
        self._batch = batch

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def resume_from_checkpoint(self, resume_path, return_checkpoint=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'], strict=self._strict_load)

        if return_checkpoint:
            return checkpoint

class TypedModel(BaseModel):
    @abstractproperty
    def type(self):
        """
        Type signature for the layer as an arrow between two vector spaces

        :return: A FirstOrderType for the model's arrow type
        """
        raise NotImplementedError()

    @abstractproperty
    def effect(self):
        """
        Effect signature for a layer as an arrow between two vector spaces

        :return: List of strings string for sampled random variables (if any)
        """
        raise NotImplementedError()

    @property
    def effects(self):
        return ', '.join(self.effect)
