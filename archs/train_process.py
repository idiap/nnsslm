"""
train_process.py

Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn

from .base import is_module_gpu
from .utils import HasNextIter

_DEFAULT_CP_FREQ = 1000

# ==== debug ==== #
# import matplotlib.pyplot as plt
# ==== debug ==== #


def _to_gpu_recursively(x):
    """
    Recursively put all tensors in x to gpu
    Args:
        x : any variable
    Returns:
        copy of x with all tensors in gpu
    """
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, list):
        return [_to_gpu_recursively(y) for y in x]
    elif isinstance(x, tuple):
        return tuple(_to_gpu_recursively(y) for y in x)
    else:
        return x


class AbstractProcess(object):
    """Abstract training process"""
    def __init__(self, optimizer, lr_scheduler, verbose=True):
        """
        Args:
            optimizer : pytorch optimizer
            lr_scheduler : a implementation of LearningRateScheduler,
                           also checkes stopping condition
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose

    def process_batch():
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        raise NotImplementedError()

    def epoch_end(self):
        """
            this method is called at the end of an epoch.
        """
        raise NotImplementedError()

    def train_end(self):
        """
            this method is called at the end of the training process.
        """
        raise NotImplementedError()

    def _prepare_optimize(self):
        """
        do anything before optimizer step
        """
        pass

    def run(self):
        batch_counter = 0
        epoch_counter = 0
        cont = True
        while cont:
            try:
                self.optimizer.zero_grad()
                loss = self.process_batch()
                loss.backward()

                # add an option to modify gradient for example
                self._prepare_optimize()

                self.optimizer.step()

                if self.verbose:
                    print('Epoch %d, Update %d, Loss: %.6f' %
                          (epoch_counter, batch_counter, loss.item()),
                          file=sys.stderr)
                batch_counter += 1

                cont, adjust, lr = self.lr_scheduler.each_batch(loss.item())
                if adjust:
                    self._adjust_lr(lr)
            except StopIteration:
                if self.verbose:
                    print('Epoch %d, END' % epoch_counter, file=sys.stderr)
                epoch_counter += 1
                self.epoch_end()

                cont, adjust, lr = self.lr_scheduler.each_epoch()
                if adjust:
                    self._adjust_lr(lr)
        self.train_end()

    def _adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if self.verbose:
                print('Adjust learning rate : %g' % param_group['lr'],
                      file=sys.stderr)


class LearningRateScheduler(object):
    def each_batch(self, loss):
        """
        update training status of this batch, and check if learning rate needs
        adjusted or stop condition is met

        Args:
            loss : mean loss of this batch

        Returns:
            cont   : False if stop condition is met
            adjust : True if learning rate needs adjustment
            lr     : new learning rate
        """
        raise NotImplementedError()

    def each_epoch(self):
        """
        update training status of this batch, and check if learning rate needs
        adjusted or stop condition is met

        Returns:
            cont   : False if stop condition is met
            adjust : True if learning rate needs adjustment
            lr     : new learning rate
        """
        raise NotImplementedError()

    def get_status(self):
        """
        get status as a string
        """
        return "get_status not implemented"


class FixedEpochs(LearningRateScheduler):
    """
    Train for a fixed number of epochs
    """
    def __init__(self, num_epochs, verbose=True):
        """
        Args:
            num_epochs : number of epochs
            verbose : print log to stderr
        """
        self.num_epochs = num_epochs
        self.verbose = verbose

        self.loss_buf_sum = 0.0
        self.batch_counter = 0
        self.epoch_counter = 0

    def each_batch(self, loss):
        self.loss_buf_sum += loss
        self.batch_counter += 1
        return True, False, self.learning_rate

    def each_epoch(self):
        loss_avg = self.loss_buf_sum / self.batch_counter
        if self.verbose:
            print('Average of epoch #%d: %.6f' %
                  (self.epoch_counter, loss_avg),
                  file=sys.stderr)
        self.loss_buf_sum = 0.0
        self.batch_counter = 0
        self.epoch_counter += 1

        cont = self.epoch_counter < self.num_epochs
        adjust, lr = self._update_lr_epoch_end()
        return cont, adjust, lr

    def get_status(self):
        return "Epoch %d/%d . %d\n" \
                % (self.epoch_counter, self.num_epochs,
                   self.batch_counter)

    def _update_lr_epoch_end(self):
        """
        Update learning rate when an epoch is finished.

        Returns:
            adjust : if learning rate changed
            lr : new learning rate
        """
        raise NotImplementedError()


class DecreaseFixedEpochs(FixedEpochs):
    """
    Decrease learning rate every fixed number of epochs.
    """
    def __init__(self,
                 init_lr,
                 num_epochs,
                 lr_decay,
                 decay_factor=0.5,
                 verbose=True):
        """
        Args:
            init_lr : intial learning rate
            num_epochs : number of epochs
            lr_decay : number of epochs for learning rate decay
            decay_factor : factor for learning rate decay
            verbose : print log to stderr
        """
        super().__init__(num_epochs, verbose=verbose)
        self.learning_rate = init_lr
        self.lr_decay = lr_decay
        self.decay_factor = decay_factor

    def _update_lr_epoch_end(self):
        if self.epoch_counter % self.lr_decay == 0:
            self.learning_rate *= self.decay_factor
            return True, self.learning_rate
        else:
            return False, self.learning_rate


class DecreaseValidLossSaturate(FixedEpochs):
    """
    Decrease learning rate when validation loss saturate
    (increase or decrease by a small factor)
    """
    def __init__(self,
                 init_lr,
                 num_epochs,
                 valid_loader,
                 net,
                 loss_func,
                 decay_factor=0.5,
                 saturate_index=1.0,
                 forward_kargs={},
                 verbose=True):
        """
        Args:
            init_lr : intial learning rate
            num_epochs : number of epochs
            num_updates_per_epoch : number of updates per epochs
            valid_loader : validation set data loader
            net : network model being trained
            loss_func : loss function
            decay_factor : factor for learning rate decay
            saturate_index : saturation occurs if the ratio between the
                             validation losses of the new epoch and old epoch
                             is greater than this index
            verbose : print log to stderr
        """
        super().__init__(num_epochs, verbose=verbose)
        self.learning_rate = init_lr
        self.valid_loader = valid_loader
        self.net = net
        self.loss_func = loss_func
        self.decay_factor = decay_factor
        self.saturate_index = saturate_index
        self.forward_kargs = forward_kargs
        self.pre_valid_loss = None

    def _update_lr_epoch_end(self):
        valid_loss = self._compute_validation_loss()
        assert valid_loss >= 0.0
        if self.pre_valid_loss is None:
            adjust = False
            ratio = np.nan
        else:
            ratio = valid_loss / self.pre_valid_loss
            adjust = ratio >= self.saturate_index
            if adjust:
                self.learning_rate *= self.decay_factor
        self.pre_valid_loss = valid_loss
        print(f'Validation Loss: {valid_loss:.3g} ({ratio:.3f})',
              file=sys.stderr)
        return adjust, self.learning_rate

    def _compute_validation_loss(self):
        # list of losses of batches
        loss_sum = 0.0
        n_samples = 0.0

        # check if gpu
        gpu = is_module_gpu(self.net)

        # put model to eval mode
        train_status = self.net.training
        self.net.eval()

        with torch.no_grad():
            for x, y in self.valid_loader:
                if gpu:
                    xv = _to_gpu_recursively(x)
                    yv = _to_gpu_recursively(y)
                else:
                    xv = x
                    yv = y

                # forward
                ov = self.net(xv, **self.forward_kargs)

                # compute loss
                loss_sum += self.loss_func(ov, yv) * xv.size(0)
                n_samples += xv.size(0)
            # average
            loss = (loss_sum / n_samples).item()
        # recover training mode
        self.net.train(train_status)
        return loss


class DecreaseOnPlateau(LearningRateScheduler):
    def __init__(self,
                 init_lr,
                 num_decay,
                 loss_buf_size=1000,
                 decay_factor=0.5,
                 verbose=True):
        self.learning_rate = init_lr
        self.num_decay = num_decay
        self.loss_buf_size = loss_buf_size
        self.decay_factor = decay_factor
        self.verbose = verbose

        self.loss_buf_sum = 0.0
        self.prev_loss_avg = None
        self.batch_counter = 0

    def each_batch(self, loss):
        self.loss_buf_sum += loss
        self.batch_counter = (self.batch_counter + 1) % self.loss_buf_size

        if self.batch_counter == 0:
            loss_avg = self.loss_buf_sum / self.loss_buf_size
            if self.verbose:
                print('Average of recent %d batches '
                      ': %.6f' % (self.loss_buf_size, loss_avg),
                      file=sys.stderr)
            self.loss_buf_sum = 0.0
            if (self.prev_loss_avg is not None
                    and loss_avg >= self.prev_loss_avg):
                self.prev_loss_avg = loss_avg
                self.num_decay -= 1
                if self.num_decay <= 0:
                    return False, False, self.learning_rate
                else:
                    self.learning_rate *= self.decay_factor
                    return True, True, self.learning_rate
            else:
                self.prev_loss_avg = loss_avg
        return True, False, self.learning_rate

    def each_epoch(self):
        return True, False, self.learning_rate

    def get_status(self):
        if self.prev_loss_avg is not None:
            return "current loss: %.6f\nremaining decays: %d\n" \
                                    % (self.prev_loss_avg, self.num_decay)
        else:
            return "current loss: unknown\nremaining decays: %d\n" \
                                    % self.num_decay


class SimpleModelProcess(AbstractProcess):
    """
        training process with one simple model
    """
    def __init__(self,
                 net,
                 model_path,
                 optimizer,
                 lr_scheduler,
                 gpu=True,
                 verbose=True):
        super().__init__(optimizer, lr_scheduler, verbose)

        if gpu:
            net.cuda()
        self.net = net
        self.model_path = model_path
        self.gpu = gpu

        # backup check point file
        self.cp_file = self.model_path + '.stat'
        if os.path.exists(self.cp_file):
            os.rename(self.cp_file, self.cp_file + '.bk')

    def _store_model(self):
        # Save model
        if self.gpu:
            self.net.cpu()
            self.net.save(self.model_path)
            self.net.cuda()
        else:
            self.net.save(self.model_path)
        self._store_status()

    def epoch_end(self):
        self._store_model()
        self.reset_data()

    def train_end(self):
        self._store_model()

    def _store_status(self):
        with open(self.cp_file, 'a') as f:
            print('#### Checkpoint ####', file=f)
            print(self.lr_scheduler.get_status(), file=f)

    def reset_data(self):
        """reset data iter"""
        raise NotImplementedError


class SupervisedProcess(SimpleModelProcess):
    """
        Standard supervised training process
    """
    def __init__(
        self,
        net,
        model_path,
        train_loader,
        criterion,
        optimizer,
        lr_scheduler,
        gpu=True,
        verbose=True,
        forward_kargs={},
        grad_clip=None,
    ):
        super().__init__(net, model_path, optimizer, lr_scheduler, gpu,
                         verbose)
        # train loader
        self.train_loader = train_loader
        self.loader_iter = iter(self.train_loader)
        self.forward_kargs = forward_kargs

        # loss
        if criterion is None:
            if gpu:
                criterion = nn.MSELoss().cuda()
            else:
                criterion = nn.MSELoss()
        self.criterion = criterion

        self.grad_clip = grad_clip
        if grad_clip is not None:
            assert grad_clip > 0.0

    def reset_data(self):
        self.loader_iter = iter(self.train_loader)

    def process_batch(self):
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        # load data
        x, y = next(self.loader_iter)

        if self.verbose:
            print(f'load batch of shape {tuple(x.shape)}', file=sys.stderr)

        # if torch tensor, put to gpu
        if self.gpu:
            xv = _to_gpu_recursively(x)
        else:
            xv = x

        # forward
        ov = self.net(xv, **self.forward_kargs)

        # ==== debug ==== #
        # plt.title('pd')
        # plt.imshow(ov.cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.title('gt')
        # plt.imshow(y.cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        # ==== debug ==== #

        # if torch tensor, put to gpu
        if self.gpu:
            yv = _to_gpu_recursively(y)
        else:
            yv = y

        loss = self.criterion(ov, yv)
        return loss

    def _prepare_optimize(self):
        # clip gradient
        if self.grad_clip is not None:
            nn.utils.clip_grad_value_(self.net.parameters(), self.grad_clip)


class AdaptationProcess(SimpleModelProcess):
    """
        Unsupervised adaptation process
    """
    def __init__(self,
                 net,
                 model_path,
                 labeled_loader,
                 adapt_loader,
                 adapt_func,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 adapt_criterion=None,
                 gpu=True,
                 verbose=True):
        super().__init__(net, model_path, optimizer, lr_scheduler, gpu,
                         verbose)
        self.adapt_func = adapt_func

        # train loader
        self.labeled_loader = labeled_loader
        self.adapt_loader = adapt_loader
        self.labeled_loader_iter = HasNextIter(self.labeled_loader)
        self.adapt_loader_iter = HasNextIter(self.adapt_loader)

        # loss
        if criterion is None:
            if gpu:
                criterion = nn.MSELoss().cuda()
            else:
                criterion = nn.MSELoss()
        if adapt_criterion is None:
            adapt_criterion = criterion
        self.criterion = criterion
        self.adapt_criterion = adapt_criterion

    def reset_data(self):
        if not self.labeled_loader_iter.has_next():
            self.labeled_loader_iter = HasNextIter(self.labeled_loader)
        if not self.adapt_loader_iter.has_next():
            self.adapt_loader_iter = HasNextIter(self.adapt_loader)

    def process_batch(self):
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        # load data
        #   - adapt data
        x, y = next(self.adapt_loader_iter)
        #   - labeled data
        ox, oy = next(self.labeled_loader_iter)

        # torch tensor to gpu
        if self.gpu:
            xv = torch.cat([x, ox]).cuda()
        else:
            xv = torch.cat([x, ox])

        # forward
        ov = self.net(xv)

        # apply unsupervised adaptation, replace ground truth y
        odata = ov.data[:len(x)]
        z = np.zeros(odata.size(), dtype='float32')
        for j in range(len(z)):
            z[j] = self.adapt_func(
                x[j].cpu().numpy(),
                odata[j].cpu().numpy(),
                y[j],
            )
        y = torch.from_numpy(z)

        # convert torch to gpu
        if self.gpu:
            yv = torch.cat([y, oy]).cuda()
        else:
            yv = torch.cat([y, oy])

        aloss = self.adapt_criterion(ov[:len(y)], yv[:len(y)])
        oloss = self.criterion(ov[len(y):], yv[len(y):])
        loss = aloss + oloss

        if self.verbose:
            print(('Loss detail : %.6f = %.6f + %.6f' %
                   (loss.item(), aloss.item(), oloss.item())),
                  file=sys.stderr)

        return loss


class AdaptDecomposedProcess(SimpleModelProcess):
    """
        Adaptation with augmentation by composition
    """
    def __init__(self,
                 net,
                 model_path,
                 labeled_loader,
                 augm_loader,
                 adapt_loader,
                 adapt_func,
                 augm_adapt_func,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 adapt_criterion=None,
                 gpu=True,
                 verbose=True,
                 fl_adapt_loader=None):
        super().__init__(net, model_path, optimizer, lr_scheduler, gpu,
                         verbose)
        self.adapt_func = adapt_func
        self.augm_adapt_func = augm_adapt_func

        # train loader
        self.labeled_loader = labeled_loader
        self.fl_adapt_loader = fl_adapt_loader
        self.augm_loader = augm_loader
        self.adapt_loader = adapt_loader
        self.labeled_loader_iter = HasNextIter(self.labeled_loader)
        if fl_adapt_loader:
            self.fl_adapt_loader_iter = HasNextIter(self.fl_adapt_loader)
        self.augm_loader_iter = HasNextIter(self.augm_loader)
        self.adapt_loader_iter = HasNextIter(self.adapt_loader)

        # loss
        if criterion is None:
            if gpu:
                criterion = nn.MSELoss().cuda()
            else:
                criterion = nn.MSELoss()
        if adapt_criterion is None:
            adapt_criterion = criterion
        self.criterion = criterion
        self.adapt_criterion = adapt_criterion

    def reset_data(self):
        if not self.labeled_loader_iter.has_next():
            self.labeled_loader_iter = HasNextIter(self.labeled_loader)
        if self.fl_adapt_loader and not self.fl_adapt_loader_iter.has_next():
            self.fl_adapt_loader_iter = HasNextIter(self.fl_adapt_loader)
        if not self.augm_loader_iter.has_next():
            self.augm_loader_iter = HasNextIter(self.augm_loader)
        if not self.adapt_loader_iter.has_next():
            self.adapt_loader_iter = HasNextIter(self.adapt_loader)

    def process_batch(self):
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        # load data
        #   - adapt data
        dx, dy = next(self.adapt_loader_iter)
        #   - augmented data
        ax, ay = next(self.augm_loader_iter)
        #   - labeled data
        ox, oy = next(self.labeled_loader_iter)
        #   - additional labeled adaptation data
        if self.fl_adapt_loader is not None:
            aox, aoy = next(self.fl_adapt_loader_iter)
            ox = torch.cat([ox, aox])
            oy = torch.cat([oy, aoy])

        # for debug
        if self.verbose:
            if self.fl_adapt_loader is None:
                print((
                    'Loaded batch: %d weakly labeled, %d augmented, %d labeled'
                    % (len(dy), len(ay), len(oy))),
                      file=sys.stderr)
            else:
                print(('Loaded batch: %d weakly labeled, '
                       '%d augmented, %d labeled,'
                       ' %d labeled adpatation' %
                       (len(dy), len(ay), len(oy) - len(aoy), len(aoy))),
                      file=sys.stderr)

        # torch tensor to gpu
        dxv = dx
        axv = ax
        oxv = ox
        if self.gpu:
            dxv = dxv.cuda()
            axv = axv.cuda()
            oxv = oxv.cuda()

        # forward
        pv = self.net(torch.cat([dxv, axv[:, 0], oxv]))

        # apply network to single (decomposed)
        self.net.eval()
        pv1 = self.net(axv[:, 1])
        pv2 = self.net(axv[:, 2])
        self.net.train()

        # apply unsupervised adaptation, replace ground truth y
        # 1. adaptation data :
        pdata = pv.data[:len(dy)]
        z = np.zeros(pdata.size(), dtype='float32')
        for j in range(len(z)):
            z[j] = self.adapt_func(None, pdata[j].cpu().numpy(), dy[j])
        dy = torch.from_numpy(z)

        # 2. augmented data : use decomposed single source segments
        pdata1 = pv1.data
        pdata2 = pv2.data
        z = np.zeros(pdata1.size(), dtype='float32')
        for j in range(len(z)):
            z[j] = self.augm_adapt_func(pdata1[j].cpu().numpy(), ay[j, 1],
                                        pdata2[j].cpu().numpy(), ay[j, 2])
        ay = torch.from_numpy(z)

        # torch tensor to gpu
        yv = torch.cat([dy, ay, oy])
        if self.gpu:
            yv = yv.cuda()

        aloss = self.adapt_criterion(pv[:len(dy) + len(ay)],
                                     yv[:len(dy) + len(ay)])
        oloss = self.criterion(pv[len(dy) + len(ay):], yv[len(dy) + len(ay):])
        loss = aloss + oloss

        if self.verbose:
            print(('Loss detail : %.6f = %.6f + %.6f' %
                   (loss.item(), aloss.item(), oloss.item())),
                  file=sys.stderr)

        return loss
