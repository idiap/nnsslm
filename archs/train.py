"""
train.py

Training procedures

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys

import torch
import torch.nn as nn

from .base import num_params
from .train_process import (DecreaseFixedEpochs, DecreaseValidLossSaturate,
                            DecreaseOnPlateau, SupervisedProcess,
                            AdaptationProcess, AdaptDecomposedProcess)


def train_nn(net,
             model,
             dataset,
             num_epochs,
             batch_size,
             learning_rate,
             lr_decay,
             save_int=False,
             verbose=True,
             gpu=True,
             criterion=None,
             shuffle=True,
             adapt_func=None,
             partial=False,
             num_lr_decay=0,
             loss_buf_size=1000,
             valid_set=None,
             saturate_index=0.999,
             forward_kargs={},
             grad_clip=None,
             batch_sampler=None,
             collate_fn=None,
             num_workers=0):
    """
    stop condition :
        if num_epochs > 0
            fixed number of epochs
        else
            until loss saturates and halve learning_rate, repeat for
            num_lr_decay times

    Args:
        save_int : deprecated
    """

    # to gpu
    if gpu:
        net.cuda()

    if verbose:
        print('Training samples: %d' % len(dataset), file=sys.stderr)
        if valid_set is not None:
            print('Validation samples: %d' % len(valid_set), file=sys.stderr)
        print(net, file=sys.stderr)
        print('# parameters: %d' % num_params(net), file=sys.stderr)

    # train loader
    if batch_sampler is None:
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    if valid_set is not None:
        if batch_sampler is not None:
            raise NotImplementedError(
                'batch sampler for validation set not implemented')
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=batch_size,
        )

    # optimizer
    if not partial:
        params = net.parameters()
    else:
        params = net.partial_params()
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if num_epochs > 0:
        if valid_set is None:
            lr_scheduler = DecreaseFixedEpochs(learning_rate,
                                               num_epochs,
                                               lr_decay,
                                               decay_factor=0.5,
                                               verbose=verbose)
        else:
            lr_scheduler = DecreaseValidLossSaturate(
                learning_rate,
                num_epochs,
                valid_loader=valid_loader,
                net=net,
                loss_func=criterion,
                saturate_index=saturate_index,
                decay_factor=0.5,
                forward_kargs=forward_kargs,
                verbose=verbose)
    else:
        lr_scheduler = DecreaseOnPlateau(learning_rate,
                                         num_lr_decay,
                                         loss_buf_size,
                                         decay_factor=0.5,
                                         valid_set=valid_set,
                                         verbose=verbose)
    proc = SupervisedProcess(net,
                             model,
                             train_loader,
                             criterion,
                             optimizer,
                             lr_scheduler,
                             gpu=gpu,
                             verbose=verbose,
                             forward_kargs=forward_kargs,
                             grad_clip=grad_clip)
    proc.run()


_STAGE1_LOSS = nn.MSELoss()


def stage1_loss(output, gt):
    ndata, ndoa, nfbank, ndelay = output.size()
    gndata, gndoa = gt.size()
    assert ndata == gndata
    assert ndoa == gndoa
    assert ndelay == 1
    egt = gt.expand((1, nfbank, ndata, ndoa)).permute(2, 3, 1, 0)
    return _STAGE1_LOSS(output, egt)


class Stage1Loss:
    """ Loss after stage one, by expanding gt along given axes
    """
    def __init__(self, loss, narrow=[]):
        self.loss = loss
        self.narrow = narrow

    def __call__(self, pred, gt):
        for dim, start, end in self.narrow:
            pred = pred.narrow(dim + 1, start, end - start)
        pndim = pred.dim()
        gndim = gt.dim()
        assert pred.size(0) == gt.size(0)
        for x in range(1, gndim):
            assert pred.size(x + pndim - gndim) == gt.size(x)
        expand = 1
        for x in range(1, pndim - gndim + 1):
            expand *= pred.size(x)

        predv = pred.contiguous().view(-1, *gt.size()[1:])
        gtv = gt.unsqueeze(1).expand(gt.size(0), expand,
                                     *gt.size()[1:]).contiguous().view(
                                         -1,
                                         *gt.size()[1:])

        return self.loss(predv, gtv)


def train_stage1(net,
                 model,
                 dataset,
                 num_epochs,
                 batch_size,
                 learning_rate,
                 lr_decay,
                 save_int,
                 verbose=True,
                 gpu=True,
                 criterion=stage1_loss,
                 grad_clip=None):
    # to gpu
    if gpu:
        net.cuda()

    if verbose:
        print('Training samples: %d' % len(dataset), file=sys.stderr)
        print(net, file=sys.stderr)
        print('# parameters: %d' % num_params(net), file=sys.stderr)

    # train loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        #                                              num_workers=8,
        pin_memory=False,
        shuffle=True)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # train
    for epoch_counter in range(num_epochs):
        # adjust learning rate
        lr = learning_rate * (0.5**(epoch_counter // lr_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            if verbose:
                print(param_group['lr'], file=sys.stderr)

        for i, (x, y) in enumerate(train_loader):
            # Convert torch tensor to
            if gpu:
                x = x.cuda()
                y = y.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net.stage1(x)

            loss = criterion(outputs, y)
            loss.backward()

            # clip gradient
            if grad_clip is not None:
                assert grad_clip > 0.0
                nn.utils.clip_grad_value_(net.parameters(), grad_clip)

            # optimize
            optimizer.step()

            if verbose:
                print(
                    f'S1 Epoch [{epoch_counter+1}/{num_epochs}], '
                    f'Step [{i+1}/{len(dataset) // batch_size}], '
                    f'Loss: {loss.item():.4f}',
                    file=sys.stderr)

        # Save the Model every epoch_counter
        if gpu:
            net.cpu()
            net.save(model)
            net.cuda()
        else:
            net.save(model)

        if save_int:
            net.save('%s_e%d' % (model, epoch_counter))


def cross_entropy_loss(out, label):
    assert out.size() == label.size()
    n, d = out.size()
    x = out.contiguous().view((n, 1, d))
    y = label.contiguous().view((n, d, 1))
    ce = torch.bmm(torch.log(x), y)
    assert ce.size() == (n, 1, 1)
    return -torch.mean(ce)


def cross_entropy_loss_2(out, label):
    nhalf = out.size()[1] / 2
    return cross_entropy_loss(out[:, :nhalf], label[:, :nhalf]) \
        + cross_entropy_loss(out[:, nhalf:], label[:, nhalf:])


class CrossEntropyLossOnSM:
    def __init__(self, weights, gpu=True):
        self.weights = torch.tensor(weights)
        if gpu:
            self.weights = self.weights.cuda()

    def __call__(self, out, label):
        logp = torch.log(torch.gather(out, 1, label.unsqueeze(1))).view(-1)
        w = self.weights[label.data]
        return (-logp * w).mean()


def adapt_nn(net,
             model,
             adapt_set,
             orig_set,
             adapt_portion,
             adapt_func,
             num_epochs,
             batch_size,
             learning_rate,
             lr_decay,
             save_int,
             verbose=True,
             gpu=True,
             criterion=None,
             partial=False,
             shuffle_adapt=True,
             shuffle_orig=True,
             adapt_crit=None,
             num_lr_decay=0,
             loss_buf_size=1000):
    # to gpu
    if gpu:
        net.cuda()

    if verbose:
        print('Adaptation samples: %d' % len(adapt_set), file=sys.stderr)
        print('Original Training samples: %d' % len(orig_set), file=sys.stderr)
        print(net, file=sys.stderr)
        print('# parameters: %d' % num_params(net), file=sys.stderr)

    # data loaders
    n_adapt = int(batch_size * adapt_portion)
    adapt_loader = torch.utils.data.DataLoader(dataset=adapt_set,
                                               batch_size=n_adapt,
                                               pin_memory=False,
                                               shuffle=shuffle_adapt)

    orig_loader = torch.utils.data.DataLoader(dataset=orig_set,
                                              batch_size=batch_size - n_adapt,
                                              pin_memory=False,
                                              shuffle=shuffle_orig)
    assert len(adapt_loader) > 0
    assert len(orig_loader) > 0

    # optimizer
    if not partial:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(net.partial_params(), lr=learning_rate)

    # train
    if num_epochs > 0:
        raise NotImplementedError()
    else:
        lr_scheduler = DecreaseOnPlateau(learning_rate,
                                         num_lr_decay,
                                         loss_buf_size,
                                         decay_factor=0.5,
                                         verbose=verbose)
    proc = AdaptationProcess(net,
                             model,
                             orig_loader,
                             adapt_loader,
                             adapt_func,
                             criterion,
                             optimizer,
                             lr_scheduler,
                             adapt_crit,
                             gpu=gpu,
                             verbose=verbose)
    proc.run()


def adapt_decomposed(net,
                     model,
                     adapt_set,
                     augm_set,
                     orig_set,
                     adapt_portion,
                     augm_portion,
                     adapt_func,
                     augm_adapt_func,
                     num_epochs,
                     batch_size,
                     learning_rate,
                     lr_decay,
                     save_int,
                     verbose=True,
                     gpu=True,
                     criterion=None,
                     partial=False,
                     shuffle_adapt=True,
                     shuffle_orig=True,
                     adapt_crit=None,
                     num_lr_decay=0,
                     loss_buf_size=1000,
                     fl_adapt_set=None,
                     fl_adapt_portion=0.0):
    assert fl_adapt_set is not None or fl_adapt_portion == 0, \
           'when fl_adapt_set is None, fl_adapt_portion should be zero'

    # to gpu
    if gpu:
        net.cuda()

    if verbose:
        print('Adaptation samples: %d' % len(adapt_set), file=sys.stderr)
        print('Original Training samples: %d' % len(orig_set), file=sys.stderr)
        print(net, file=sys.stderr)
        print('# parameters: %d' % num_params(net), file=sys.stderr)

    # data loaders
    n_adapt = int(batch_size * adapt_portion)
    n_augm = int(batch_size * augm_portion)
    n_fl_adapt = int(batch_size * fl_adapt_portion)
    n_orig = batch_size - n_adapt - n_augm - n_fl_adapt
    assert n_orig > 0

    adapt_loader = torch.utils.data.DataLoader(dataset=adapt_set,
                                               batch_size=n_adapt,
                                               pin_memory=False,
                                               shuffle=shuffle_adapt)

    augm_loader = torch.utils.data.DataLoader(dataset=augm_set,
                                              batch_size=n_augm,
                                              pin_memory=False,
                                              shuffle=shuffle_adapt)

    orig_loader = torch.utils.data.DataLoader(dataset=orig_set,
                                              batch_size=n_orig,
                                              pin_memory=False,
                                              shuffle=shuffle_orig)

    if fl_adapt_set is not None:
        fl_adapt_loader = torch.utils.data.DataLoader(dataset=orig_set,
                                                      batch_size=n_fl_adapt,
                                                      pin_memory=False,
                                                      shuffle=True)
    else:
        fl_adapt_loader = None
    assert len(adapt_loader) > 0
    assert len(augm_loader) > 0
    assert len(orig_loader) > 0

    # optimizer
    if not partial:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(net.partial_params(), lr=learning_rate)

    # train
    if num_epochs > 0:
        raise NotImplementedError()
    else:
        lr_scheduler = DecreaseOnPlateau(learning_rate,
                                         num_lr_decay,
                                         loss_buf_size,
                                         decay_factor=0.5,
                                         verbose=verbose)
    proc = AdaptDecomposedProcess(net,
                                  model,
                                  orig_loader,
                                  augm_loader,
                                  adapt_loader,
                                  adapt_func,
                                  augm_adapt_func,
                                  criterion,
                                  optimizer,
                                  lr_scheduler,
                                  adapt_crit,
                                  gpu=gpu,
                                  verbose=verbose,
                                  fl_adapt_loader=fl_adapt_loader)
    proc.run()


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
