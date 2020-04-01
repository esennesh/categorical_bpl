import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import FirstOrderType, TypedModel

class EncInputLayer(TypedModel):
    def __init__(self, input_dim=28*28):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
        )
        self._input_dim = input_dim

    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, torch.Size([self._input_dim])),
            FirstOrderType.TENSORT(torch.float, torch.Size([512]))
        )

    def forward(self, inputs):
        return self.layer(inputs)

class NonreductiveLayer(TypedModel):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
        )

    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, torch.Size([512])),
            FirstOrderType.TENSORT(torch.float, torch.Size([512]))
        )

    def forward(self, inputs):
        return self.layer(inputs)

class ReductiveLayer(TypedModel):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )

    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, torch.Size([512])),
            FirstOrderType.TENSORT(torch.float, torch.Size([256]))
        )

    def forward(self, inputs):
        return self.layer(inputs)

class EncOutputLayer(TypedModel):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.layer = nn.Linear(256, hidden_dim * 2)
        self._out_dim = hidden_dim * 2

    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, torch.Size([256])),
            FirstOrderType.TENSORT(torch.float, torch.Size([self._out_dim]))
        )

    def forward(self, inputs):
        return self.layer(inputs)

class DiagonalGaussianLayer(TypedModel):
    def __init__(self, layer_class, latent_name=None):
        super().__init__()
        self.layer = layer_class()
        l, r = self.layer.type().arrowt()
        _, self._in_dim = l.tensort()
        _, self._out_dim = r.tensort()
        assert self._out_dim[0] % 2 == 0
        self._out_dim = torch.Size([self._out_dim[0] // 2])
        if not latent_name:
            latent_name = 'Z^{%d -> %d}' % (self._in_dim[0], self._out_dim[0])
        self._latent_name = latent_name

    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TENSORT(torch.float, self._in_dim),
            FirstOrderType.TENSORT(torch.float, self._out_dim)
        )

    def forward(self, inputs):
        with name_count():
            zs = self.layer(inputs).view(-1, 2, self._out_dim)
            normal = dist.Normal(zs[:, 0], F.softplus(zs[:, 1])).to_event(1)
            return pyro.sample(self._latent_name, normal)

class StandardNormalLayer(TypedModel):
    def __init__(self, dim, latent_name=None):
        super().__init__()
        if not latent_name:
            latent_name = 'Z^{N(0, 1)}_{%d}' % dim
        self._latent_name = latent_name
        self._dim = dim

    def type(self):
        return FirstOrderType.ARROWT(
            FirstOrderType.TOPT(),
            FirstOrderType.TENSORT(torch.float, torch.Size([self._dim]))
        )

    def forward(self, inputs):
        with name_count():
            z_loc = inputs.new_zeros(torch.Size((inputs.shape[0], self._dim)))
            z_scale = inputs.new_ones(torch.Size((inputs.shape[0], self._dim)))
            normal = dist.Normal(z_loc, z_scale).to_event(1)
            return pyro.sample(self._latent_name, normal)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
