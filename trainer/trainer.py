import numpy as np
from pyro.infer import SVI, TraceGraph_ELBO
from pyro.optim import Adam, PyroOptim
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import ImportanceSampler, inf_loop, MetricTracker

class EmCounter:
    def __init__(self, estep_params, mstep_params, epochs=10, even_estep=True):
        self._params = {
            'expectation': set(estep_params),
            'maximization': set(mstep_params),
        }
        self._epochs = epochs
        self._even_estep = even_estep
        self._counter = 0

    def get_state(self):
        return {
            'params': self._params,
            'epochs': self._epochs,
            'even_estep': self._even_estep,
            'counter': self._counter
        }

    def set_state(self, state_dict):
        self._params = state_dict['params']
        self._epochs = state_dict['epochs']
        self._even_estep = state_dict['even_estep']
        self._counter = state_dict['counter']

    def step_params(self, params):
        self._counter += 1
        if self._counter % self._epochs == 0:
            step = 'expectation' if self._even_estep else 'maximization'
        else:
            step = 'maximization' if self._even_estep else 'expectation'
        return self._params[step] & set(params)

class ExpectationMaximizationOptim(PyroOptim):
    def __init__(self, counter, optim_constructor, optim_args, clip_args=None):
        assert isinstance(counter, EmCounter)
        self._counter = counter
        super().__init__(optim_constructor, optim_args, clip_args)

    def __call__(self, params, *args, **kwargs):
        params = self._counter.step_params(params)
        super().__call__(params, *args, **kwargs)

    def get_state(self):
        state_dict = super().get_state()
        state_dict['_counter'] = self._counter.get_state()
        return state_dict

    def set_state(self, state_dict):
        self._counter.set_state(state_dict['_counter'])
        super().set_state(state_dict)

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 jit=False, log_images=True):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_length))
        self.jit = jit
        self.log_images = log_images

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'log_likelihood', 'log_marginal',
                                           *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        elbo = TraceGraph_ELBO(vectorize_particles=False, num_particles=4)
        svi = SVI(self.model.model, self.model.guide, self.optimizer, loss=elbo)

        self.model.train()
        self.train_metrics.reset()
        current = 0
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            loss = svi.step(observations=data) / data.shape[0]

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            current += len(target)
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, current=current),
                    loss))
                if self.log_images:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log['loss'])
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        elbo = TraceGraph_ELBO(vectorize_particles=False, num_particles=4)
        svi = SVI(self.model.model, self.model.guide, self.optimizer, loss=elbo)
        imps = ImportanceSampler(self.model.model, self.model.guide,
                                 num_samples=4)

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                loss = svi.evaluate_loss(observations=data) / data.shape[0]
                imps.sample(observations=data)
                log_likelihood = imps.get_log_likelihood().item() / data.shape[0]
                log_marginal = imps.get_log_normalizer().item() / data.shape[0]

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss)
                self.valid_metrics.update('log_likelihood', log_likelihood)
                self.valid_metrics.update('log_marginal', log_marginal)

                if self.log_images:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return self.valid_metrics.result()

    def _progress(self, batch_idx, current=None):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            if current is None:
                current = batch_idx * self.data_loader.batch_length
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
