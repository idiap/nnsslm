"""
adaptation.py

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import math
import itertools

import numpy as np
import torch

_DEGREE = 1.0 / 180.0 * math.pi
_MIN_AZIMUTH = -math.pi
_MAX_AZIMUTH = math.pi


def _l2_distance(a, b):
    assert a.shape == b.shape
    return np.linalg.norm(a - b)


def _search_grid(pred, grid, prepare_gt, fdist):
    min_dist = float('inf')
    best_m = None
    best_doas = None

    for doas in itertools.product(*grid):
        y = prepare_gt([((math.cos(d), math.sin(d), 0), 1, None)
                        for d in doas])
        dist = fdist(pred, y)
        if dist < min_dist:
            min_dist = dist
            best_m = y
            best_doas = doas
    return min_dist, best_m, best_doas


def _nogs_fixed_nsrc(pred,
                     nsrc,
                     prepare_gt,
                     resolution=np.array([8, 4, 2, 1, .5, .25]) * _DEGREE,
                     fdist=_l2_distance):
    """Nearest output gird search : fixed number of sources

    Args:
        pred       : predicted output
        nsrc       : number of sources
        prepare_gt : function for output coding
        resolution : resolutions for grid search
        fdist      : function for distance measure

    Returns:
        dist  : distance to the best match
        match : best matching output code
    """
    if nsrc == 0:
        match = prepare_gt([])
        dist = fdist(pred, match)
    else:
        assert nsrc > 0

        # initial grid
        grid = [
            np.linspace(
                _MIN_AZIMUTH, _MAX_AZIMUTH,
                math.ceil((_MAX_AZIMUTH - _MIN_AZIMUTH) / resolution[0]) + 1)
            for _ in range(nsrc)
        ]
        dist, match, doas = _search_grid(pred, grid, prepare_gt, fdist)

        pr = resolution[0]
        for nr in resolution[1:]:
            grid = [
                np.linspace(d - pr, d + pr,
                            math.ceil(2 * pr / nr) + 1) for d in doas
            ]

            dist, match, doas = _search_grid(pred, grid, prepare_gt, fdist)
            pr = nr
    return dist, match


class NearestOutputGridSearch(object):
    """
    Match the predicted output to the nearest possible (correct) output
    with grid search.

    The search grid only considers (speech) sources on the horizontal plane
    """
    def __init__(self,
                 prepare_gt,
                 possible_nsrc,
                 resolution=np.array([8, 4, 2, 1, .5, .25]) * _DEGREE):
        """
        Args:
            prepare_gt    : function for output coding
            possible_nsrc : list of possible number of sources
            resolution    : resolutions for grid search
        """
        self.prepare_gt = prepare_gt
        self.possible_nsrc = possible_nsrc
        self.resolution = resolution

    def __call__(self, indata, outpred, outgt):
        # ignore indata and outgt
        min_dist = float('inf')
        best_m = None

        for nsrc in self.possible_nsrc:
            dist, m = _nogs_fixed_nsrc(outpred, nsrc, self.prepare_gt,
                                       self.resolution)
            if dist < min_dist:
                min_dist = dist
                best_m = m

        return best_m


def _volume(stft):
    r = stft[:4]
    i = stft[4:]
    return np.mean(r * r + i * i)


class VolumeBasedNearestOutput(object):
    """
    Match the predicted output to the nearest possible (correct) output
    with grid search. The possible output is selected based on volume.

    If the volume is higher than a threshold, there is at least one source.

    The search grid only considers (speech) sources on the horizontal plane
    """
    def __init__(self,
                 prepare_gt,
                 max_nsrc,
                 act_threshold,
                 resolution=np.array([8, 4, 2, 1, .5, .25]) * _DEGREE):
        """
        Args:
            prepare_gt    : function for output coding
            max_nsrc      : maximum possible number of sources
            resolution    : resolutions for grid search
            act_threshold : volumue threshold for activeness in dB
                            (at least one source)
        """
        self.prepare_gt = prepare_gt
        self.max_nsrc = max_nsrc
        self.resolution = resolution
        self.pow_th = np.power(10.0, act_threshold / 10.0)

    def __call__(self, indata, outpred, outgt):
        # check if active
        if _volume(indata) >= self.pow_th:
            possible_nsrc = range(1, self.max_nsrc + 1)
        else:
            possible_nsrc = range(self.max_nsrc + 1)

        # ignore indata and outgt
        min_dist = float('inf')
        best_m = None

        for nsrc in possible_nsrc:
            dist, m = _nogs_fixed_nsrc(outpred, nsrc, self.prepare_gt,
                                       self.resolution)
            if dist < min_dist:
                min_dist = dist
                best_m = m

        return best_m


class KnownNSrcNearestOutput(object):
    """
    Match the predicted output to the nearest possible (correct) output
    with grid search. The possible output is selected based on the knowledge of
    number of sources.

    The search grid only considers (speech) sources on the horizontal plane
    """
    def __init__(self,
                 prepare_gt,
                 resolution=np.array([8, 4, 2, 1, .5, .25]) * _DEGREE):
        """
        Args:
            prepare_gt    : function for output coding
            max_nsrc      : maximum possible number of sources
            resolution    : resolutions for grid search
            act_threshold : volumue threshold for activeness in dB
                            (at least one source)
        """
        self.prepare_gt = prepare_gt
        self.resolution = resolution

    def __call__(self, indata, outpred, outgt):
        # known number of sources
        nsrc = outgt
        dist, m = _nogs_fixed_nsrc(outpred, nsrc, self.prepare_gt,
                                   self.resolution)
        return m


class DecomposedAdapt:
    def __init__(self, adapt_func):
        self.adapt_func = adapt_func

    def __call__(self, outpred1, outgt1, outpred2, outgt2):
        y1 = self.adapt_func(None, outpred1, outgt1)
        y2 = self.adapt_func(None, outpred2, outgt2)
        return np.maximum(y1, y2)


class ModifiedLoss:
    def __init__(self, loss, alpha, beta, reduction='mean'):
        """ modified loss = (loss + alpha) ^ beta - alpha ^ beta

        Args:
            loss      : origi# nal loss function, without reduction (sum or mean)
            alpha     : parameter
            beta      : parameter
            reduction : (default 'mean') reduction method
                        'mean', 'sum' or 'none'
        """
        self.loss = loss
        self.alpha = alpha
        self.beta = beta
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            self.reduce = lambda x: x

    def __call__(self, pred, gt):
        L = self.loss(pred, gt)
        assert len(L) == len(pred)  # reduction after modification
        mL = (L + self.alpha)**self.beta - self.alpha**self.beta
        return self.reduce(mL)


def fully_supervised(indata, outpred, outgt):
    return outgt


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
