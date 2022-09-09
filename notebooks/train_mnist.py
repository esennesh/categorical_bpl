#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '..')


# In[2]:


import argparse
import collections
import pyro
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# pyro.enable_validation(True)
# torch.autograd.set_detect_anomaly(True)


# In[5]:


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# In[6]:


Args = collections.namedtuple('Args', 'config resume device')
config = ConfigParser.from_args(Args(config='mnist_config.json', resume=None, device=None))


# In[7]:


logger = config.get_logger('train')


# In[8]:


# setup data_loader instances
data_loader = config.init_obj('data_loader', module_data)
valid_data_loader = data_loader.split_validation()


# In[9]:


# build model architecture, then print to console
model = config.init_obj('arch', module_arch)


# In[10]:


optimizer = pyro.optim.ReduceLROnPlateau({
    'optimizer': torch.optim.Adam,
    'optim_args': {
        "lr": 1e-3,
        "weight_decay": 0,
        "amsgrad": True
    },
    "patience": 50,
    "cooldown": 25,
    "factor": 0.1,
    "verbose": True,
})


# In[11]:


# optimizer = config.init_obj('optimizer', pyro.optim)


# In[12]:


trainer = Trainer(model, [], optimizer, config=config,
                  data_loader=data_loader,
                  valid_data_loader=valid_data_loader,
                  lr_scheduler=optimizer)


# In[ ]:


trainer.train()


# In[ ]:


model.cpu()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


for k in range(10):
    path, sample = model(None)
    sample = sample.view(28, 28).detach().cpu().numpy()
    path.draw()

    plt.title('Sample from prior')
    plt.imshow(sample)
    plt.show()


# In[ ]:




