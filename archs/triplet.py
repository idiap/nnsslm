"""
triplet.py

Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys

import torch

from .base import num_params


def _to_variable(x, gpu):
    if isinstance(x, torch.Tensor):
        v = x
        if gpu:
            v = v.cuda()
    else:
        v = x
    return v


def train_multitask_triplet(net,
                            model,
                            triplet_loader,
                            criterion,
                            num_epochs,
                            learning_rate,
                            lr_decay,
                            save_int,
                            verbose=True,
                            gpu=True,
                            partial=False):
    # to gpu
    if gpu:
        net.cuda()

    if verbose:
        print('Training samples: %d batches' % len(triplet_loader),
              file=sys.stderr)
        print(net, file=sys.stderr)
        print('# parameters: %d' % num_params(net), file=sys.stderr)

    # optimizer
    if not partial:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(net.partial_params(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        # adjust learning rate
        lr = learning_rate * (0.5**(epoch // lr_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            if verbose:
                print(param_group['lr'], file=sys.stderr)

        for i, ((anc_x, anc_y), (pos_x, pos_y), (neg_x, neg_y)) \
                                                in enumerate(triplet_loader):
            # Convert to Variable
            anc_xv = _to_variable(anc_x, gpu)
            pos_xv = _to_variable(pos_x, gpu)
            neg_xv = _to_variable(neg_x, gpu)

            anc_yv = [_to_variable(yt, gpu) for yt in anc_y]
            pos_yv = [_to_variable(yt, gpu) for yt in pos_y]
            neg_yv = [_to_variable(yt, gpu) for yt in neg_y]

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            anc_ov = net(anc_xv)
            pos_ov = net(pos_xv)
            neg_ov = net(neg_xv)

            # compute loss as triplets
            loss = criterion((anc_ov, pos_ov, neg_ov),
                             (anc_yv, pos_yv, neg_yv))

            loss.backward()
            optimizer.step()

            # output status
            if verbose:
                print('Epoch [%d/%d], Triplets [%d/%d], ' \
                         'Loss: %.4f' % (epoch+1, num_epochs, i,
                                         len(triplet_loader), loss.item()), file=sys.stderr)

        # Save the Model every epoch
        if gpu:
            net.cpu()
            net.save(model)
            net.cuda()
        else:
            net.save(model)

        if save_int:
            net.save('%s_e%d' % (model, epoch))


class MultitaskLossOnTriplet(object):
    def __init__(self, losses, weight=None, verbose=True):
        self.losses = losses
        self.weight = weight if weight is not None else [1.0 for _ in losses]
        self.verbose = verbose

    def __call__(self, output, ground):
        anc_ov, pos_ov, neg_ov = output
        anc_yv, pos_yv, neg_yv = ground

        ll = [
            l((ao, po, no), (ay, py, ny)) for l, ao, po, no, ay, py, ny in zip(
                self.losses, anc_ov, pos_ov, neg_ov, anc_yv, pos_yv, neg_yv)
        ]
        if self.verbose:
            print('Loss on each task : [%s]' \
                                        % ', '.join('%.3g' % x.item() for x in ll), file=sys.stderr)
        return sum(l * w for l, w in zip(ll, self.weight))


class TripletLoss(object):
    def __init__(self, alpha, metric, preproc=(lambda x, y: x)):
        self.alpha = alpha
        self.metric = metric
        self.preproc = preproc

    def __call__(self, output, ground):
        """
        Args:
            output : network output (prediction) 
                     as triplets : anchor, positive, negative
                     each item is a list of embeddings
            ground : not used
        """
        ao, po, no = output
        ay, py, ny = ground
        a = self.preproc(ao, ay)
        p = self.preproc(po, py)
        n = self.preproc(no, ny)
        margin = self.metric(a, p) - self.metric(a, n) + self.alpha
        return torch.mean(torch.clamp(margin, min=0.0))


class LossExpandToTriplet(object):
    def __init__(self, loss):
        """
        Args:
            loss : original loss function
        """
        self.loss = loss

    def __call__(self, output, ground):
        ao, po, no = output
        ay, py, ny = ground
        return (self.loss(ao, ay) + self.loss(po, py) +
                self.loss(no, ny)) / 3.0


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
