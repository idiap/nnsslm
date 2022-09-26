"""
dann.py

Domain-Adverserial Neural Network

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from .base import num_params


class GradReversal(Function):
    """ Gradient Reversal Layer """
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return -grad_output


class DANN(nn.Module):
    """ Domain-Adverserial Neural Network """
    def __init__(self, forward_net, domain_classifier):
        """
        Args:
            forward_net : forward network, feature extractor + label predictor
                          the network implements forward_feature()
            domain_classifier : as its name
        """
        super().__init__()
        self.fn = forward_net
        self.dc = domain_classifier
        self.grl = GradReversal()

    def forward(self, x):
        return self.fn.forward(x)

    def forward_domain(self, x):
        f = self.fn.forward_feature(x)
        return self.dc.forward(self.grl(f))


def train_dann(forward_net,
               domain_classifier,
               fn_name,
               dc_name,
               fn_loss,
               dc_loss,
               target_set,
               source_set,
               num_epochs,
               batch_size,
               learning_rate,
               lr_decay,
               lambda_=1.0,
               target_portion=0.5,
               save_int=False,
               verbose=True,
               gpu=True,
               nbatch_per_epoch=None,
               shuffle_target=True,
               shuffle_source=True,
               adapt_func=None,
               varying_lambda=False):
    # to gpu
    if gpu:
        forward_net.cuda()
        domain_classifier.cuda()

    if verbose:
        print('Target samples: %d' % len(target_set), file=sys.stderr)
        print('Source samples: %d' % len(source_set), file=sys.stderr)
        print('Foward network:', file=sys.stderr)
        print('# parameters: %d' % num_params(forward_net), file=sys.stderr)
        print(forward_net, file=sys.stderr)
        print('Domain classifier:', file=sys.stderr)
        print('# parameters: %d' % num_params(domain_classifier),
              file=sys.stderr)
        print(domain_classifier, file=sys.stderr)

    # data loaders
    n_target = int(batch_size * target_portion)
    target_loader = torch.utils.data.DataLoader(dataset=target_set,
                                                batch_size=n_target,
                                                pin_memory=False,
                                                shuffle=shuffle_target)

    source_loader = torch.utils.data.DataLoader(dataset=source_set,
                                                batch_size=batch_size -
                                                n_target,
                                                pin_memory=False,
                                                shuffle=shuffle_source)
    assert len(target_loader) > 0
    assert len(source_loader) > 0

    # initiate dann
    net = DANN(forward_net, domain_classifier)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # train
    source_it = iter(source_loader)
    for epoch in range(num_epochs):
        # adjust learning rate
        lr = learning_rate * (0.5**(epoch // lr_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            if verbose:
                print(param_group['lr'], file=sys.stderr)

        for i, (x, y) in enumerate(target_loader):
            if nbatch_per_epoch is not None and i >= nbatch_per_epoch:
                break

            # load source data as well
            try:
                sx, sy = next(source_it)
            except StopIteration:
                source_it = iter(source_loader)
                sx, sy = next(source_it)

            # Convert torch tensor to Variable
            dxv = torch.cat([x, sx])
            if adapt_func is not None:
                fxv = dxv
            else:
                fxv = sx

            # convert to cuda if needed
            if gpu:
                fxv = fxv.cuda()
                dxv = dxv.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            fov = net(fxv)
            dov = net.forward_domain(dxv)

            # forward net ground truth
            if adapt_func is not None:
                # apply adaptation
                tdata = fov.data[:len(x)]
                ty = np.zeros(tdata.size(), dtype='float32')
                for j in range(len(ty)):
                    ty[j] = adapt_func(x[j].cpu().numpy(),
                                       tdata[j].cpu().numpy(), y[j])
                ty = torch.from_numpy(ty)
                fyv = torch.cat([ty, sy])
            else:
                fyv = sy

            # get domain ground truth
            dy = torch.cat([torch.ones(len(x)), torch.zeros(len(sx))]).long()
            dyv = dy

            # convert to cuda if needed
            if gpu:
                fyv = fyv.cuda()
                dyv = dyv.cuda()

            # compute the runtime lambda
            if varying_lambda:
                prog = (epoch +
                        1.0 * i * n_target / len(target_set)) / num_epochs
                lp = lambda_ * (2.0 / (1.0 + math.exp(-10.0 * prog)) - 1.0)
            else:
                lp = lambda_

            fl = fn_loss(fov, fyv)
            dl = dc_loss(dov, dyv)
            loss = fl + lp * dl
            loss.backward()
            optimizer.step()

            if verbose:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f = %.4f + %.1g * %.4f' \
                    % (epoch+1, num_epochs, i+1, len(target_set) // n_target,
                       loss.item(), fl.item(), lp, dl.item()), file=sys.stderr)

        # Save the Model every epoch
        if gpu:
            net.cpu()
            forward_net.save(fn_name)
            domain_classifier.save(dc_name)
            net.cuda()
        else:
            forward_net.save(fn_name)
            domain_classifier.save(dc_name)

        if save_int:
            forward_net.save('%s_e%d' % (fn_name, epoch))
            domain_classifier.save('%s_e%d' % (dc_name, epoch))


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
