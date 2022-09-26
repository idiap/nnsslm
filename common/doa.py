"""
doa.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import torch

import apkit

from .datasets import _gt_azimuth_distance, _gt_yaxis_symm_distance

MAX_DOA_DISTANCE = np.pi


def sample_azimuth_and_get_metric(n_doa, yaxis_symm=False):
    """
    Args:
        n_doa : number of directions
        yaxis_symm :
            If the microphone array is linear (e.g. kinect),
            the localization is sysmmetric w.r.t. an axis.
            Here, we assume it is w.r.t the y-axis.
    Returns:
        aindex : sampled DOAs (azimuth)
        dmetric : distance metric
    """
    if not yaxis_symm:
        aindex = np.arange(n_doa) * 2 * np.pi / n_doa
        dmetric = _gt_azimuth_distance
    else:
        aindex = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_doa)
        dmetric = _gt_yaxis_symm_distance
    return aindex, dmetric


def sample_azimuth_3d(n_doa, yaxis_symm=False):
    """
    Args:
        n_doa : number of directions
        yaxis_symm :
            If the microphone array is linear (e.g. kinect),
            the localization is sysmmetric w.r.t. an axis.
            Here, we assume it is w.r.t the y-axis.
    Returns:
        sampled directions in 3d (x, y, z), which norm is 1.0
    """
    aindex, _ = sample_azimuth_and_get_metric(n_doa, yaxis_symm)
    return np.array([np.cos(aindex), np.sin(aindex), np.zeros(n_doa)]).T


def loc_to_id(loc, ref_doas, metric=apkit.angular_distance):
    """
    Get DOA ID of a sound source location

    Args:
        loc : coordinate of a sound source location, tuple of (x, y, z)
        ref_doas : list of reference DOAs, list of (x, y, z)

    Returns:
        DOA ID, the ID such that ref_doas[ID] is closest to the loc
    """
    _, min_id = min((metric(loc, r), i) for i, r in enumerate(ref_doas))
    return min_id


class AbstractDOAwiseGT(object):
    """
    Abstract class for all DOA-wise ground truth loader.
    """
    def __init__(self,
                 n_doa,
                 yaxis_symm=False,
                 src_filter=None,
                 dtype='float32'):
        """
        Args:
            n_doa : number of directions
            yaxis_symm :
                If the microphone array is linear (e.g. kinect),
                the localization is sysmmetric w.r.t. an axis.
                Here, we assume it is w.r.t the y-axis.
            src_filter : a function to select a subset of the sound sources
            dtype : data type of grund truth tensor
        """
        self.n_doa = n_doa
        self.src_filter = src_filter
        self.dtype = dtype
        self.aindex, self.dmetric = sample_azimuth_and_get_metric(
            n_doa,
            yaxis_symm,
        )

    def __call__(self, gt_srcs):
        """
        Load ground truth tensor.
        Subclass should implement (override) _internal_call

        Args:
            gt_srcs : ground truth sources,
                      list of (loc, source_type, speaker)

        Returns:
            gt_tensor : ground truth tensor of the desired type
        """
        # apply filter
        if self.src_filter is not None:
            gt_srcs = [src for src in gt_srcs if self.src_filter(src)]

        # call actual loader
        gt_tensor = self._internal_call(gt_srcs)

        # convert type
        return gt_tensor.astype(self.dtype, copy=False)

    def _internal_call(self, gt_srcs):
        """
        Subclass should implement (override) _internal_call

        Args:
            gt_srcs : filtered ground truth sources,
                      list of (loc, source_type, speaker)

        Returns:
            gt_tensor : ground truth tensor
        """
        raise NotImplementedError()

    def _get_distance_matrix(self, gt_srcs):
        """
        Get distance matrix

        Args:
            gt_srcs : ground truth sources,
                      list of (loc, source_type, speaker)

        Returns:
            dmat : distance between ground truth sources and reference DOAs,
                   array of (# sources, # doas)
        """
        # distance matrix
        dmat = np.zeros((len(gt_srcs), len(self.aindex)))
        for j, (loc, _, _) in enumerate(gt_srcs):
            dmat[j] = self.dmetric(loc, self.aindex)
        return dmat


class GaussianSpatialSpectrum(AbstractDOAwiseGT):
    """
    Guassian-shaped spatial spectrum as ground truth for DOA estimation
    """
    def __init__(self,
                 n_doa,
                 sigma,
                 yaxis_symm=False,
                 src_filter=None,
                 dtype='float32'):
        """
        Args:
            n_doa : number of directions
            sigma : parameter for curve width (radian)
            yaxis_symm : symmetric w.r.t y-axis
                         (see doa.AbstractDOAwiseGT)
            src_filter : a function to select a subset of the sound sources
            dtype : data type of grund truth tensor
        """
        super().__init__(n_doa, yaxis_symm, src_filter, dtype)
        self.sigma = sigma

    def _internal_call(self, gt_srcs):
        """
        Args:
            gt_srcs : ground truth sources,
                      list of (loc, source_type, speaker)
        """
        # ground truth spatial spectrum
        gt_tensor = np.zeros(self.n_doa)

        for d in self._get_distance_matrix(gt_srcs):
            # maximum of the Guassian curves
            gt_tensor = np.maximum(gt_tensor, np.exp(-d**2 / self.sigma**2))

        return gt_tensor


class TriangleSpatialSpectrum(AbstractDOAwiseGT):
    """
    Triangle-shaped spatial spectrum as ground truth for DOA estimation

    The spatial spectrum values depend on their distance to the target sources
    and the interference sources. Let A be distance to the target source, and B
    be the distance to the nearest interferance. The value is:
      | max(1.0 - A / min(B, sigma), 0.0), if B > 0
      | 0.0,                               otherwise (A = B = 0)
    """
    def __init__(self,
                 n_doa,
                 sigma,
                 yaxis_symm=False,
                 src_filter=None,
                 dtype='float32'):
        """
        Args:
            n_doa : number of directions
            sigma : neighourhood size
            yaxis_symm : symmetric w.r.t y-axis
                         (see doa.AbstractDOAwiseGT)
            src_filter : a function to select a subset of the sound sources
            dtype : data type of grund truth tensor
        """
        super().__init__(n_doa, yaxis_symm, src_filter, dtype)
        self.sigma = sigma

    def _internal_call(self, gt_srcs):
        """
        Args:
            gt_srcs : ground truth sources,
                      list of (loc, source_type, speaker)
        """
        # ground truth spatial spectrum
        gt_tensor = np.zeros(self.n_doa)

        if len(gt_srcs) > 0:
            # distance matrix
            dmat = self._get_distance_matrix(gt_srcs)

            # source id of the target source
            target_srcid = np.argmin(dmat, axis=0)

            # distance to the target source (A)
            target_dist = np.choose(target_srcid, dmat)

            # distance to the interference (B) : minimum sigma
            if len(gt_srcs) > 1:
                # find the second closest source
                # set distance to target source to very big
                np.put_along_axis(
                    dmat,
                    np.expand_dims(target_srcid, axis=0),
                    MAX_DOA_DISTANCE + 1.0,
                    axis=0,
                )
                inter_dist = np.amin(dmat, axis=0)
                inter_dist = np.minimum(self.sigma, inter_dist)
            else:
                inter_dist = self.sigma

            # result
            gt_tensor = np.maximum(
                1.0 - (target_dist + 1e-10) / (inter_dist + 1e-10),
                0.0,
            )

        return gt_tensor


def _broadcast_to(x, target_size, axis):
    if x is None:
        return None
    # check
    xdim = x.dim()
    assert x.shape[:axis] == target_size[:axis]
    if axis < xdim:
        assert x.shape[axis:] == target_size[-(xdim - axis):]
    for _ in range(len(target_size) - xdim):
        # fill missing axes
        x = x.unsqueeze(axis)
    return x


class DOAwiseMSELoss(object):
    """
    DOA-wise MSE loss.
    Expand to time axis if needed.
    """
    def __init__(self, encoder=None):
        """
        Args:
            encoder : function to encode the ground truth.
                      None, if the input gt is already encoded,
        """
        self.encoder = encoder

    def __call__(self, pred, gt):
        """
        Args:
            pred : prediction of the spatial spectra,
                   tensor of (sample, doa) or (sample, time, doa)
            gt : ground truth spatial spectrum, tensor of (sample, doa)
                 OR tuple of (gt_tensor, weight)
                 OR raw labels (to be encoded by encoder)

        Returns:
            loss
        """
        if self.encoder is not None:
            gt = self.encoder(gt)

        if type(gt) == tuple or type(gt) == list:
            gt_tensor, weight = gt
        else:
            gt_tensor, weight = gt, None

        # broadcast gt and weight to dim of pred
        gt_tensor_bc = _broadcast_to(gt_tensor, pred.size(), axis=1)
        weight_bc = _broadcast_to(weight, pred.size(), axis=2)

        # MSE loss between prediction and ground truth spatial spectra
        if weight is None:
            return torch.mean((pred - gt_tensor_bc)**2.0)
        else:
            return torch.mean((pred - gt_tensor_bc)**2.0 * weight_bc)


class GTLoaderWithWeight(object):
    """
    Combine ground truth loader with extra weighting values
    """
    def __init__(self, gt_loader, weight_loader):
        """
        Args:
            gt_loader : ground truth loader
            weight_loader : weight loader
        """
        self.gt_loader = gt_loader
        self.weight_loader = weight_loader

    def __call__(self, gt_srcs):
        """
        Load ground truth and weight tensors

        Args:
            gt_srcs : ground truth sources,
                      list of (loc, source_type, speaker)

        Returns:
            gt_tensor : ground truth tensor
            weight : weight tensor
        """
        return self.gt_loader(gt_srcs), self.weight_loader(gt_srcs)


class SingleSourceGaussianEncoder(object):
    """
    Encoder for DOAwiseMSELoss.
    Single-source Gaussian spatial spectrum
    """
    def __init__(self,
                 n_doa,
                 sigma,
                 circular=True,
                 device=None,
                 dtype='float32'):
        """
        Args:
            n_doa : number of directions
            sigma : parameter for curve width (unit is sampled directions)
        """
        self.n_doa = n_doa
        self.sigma = sigma
        self.circular = circular
        self.device = device
        self.dtype = dtype

    def __call__(self, l_doa):
        """
        Args:
            y : prediction, torch.Tensor of (samples, direction)
            l_doa : list of ground truth doa label (index of dir)
        """
        n_samples = len(l_doa)
        a = torch.arange(self.n_doa, dtype=self.dtype,
                         device=self.device).unsqueeze(0)
        l_doa = torch.tensor(l_doa, dtype=self.dtype,
                             device=self.device).unsqueeze(1)
        raw_d = a - l_doa
        assert raw_d.size() == (n_samples, self.n_doa)
        d = torch.abs(raw_d)
        if self.circular:
            d = torch.min(d, torch.abs(raw_d + self.n_doa))
            d = torch.min(d, torch.abs(raw_d - self.n_doa))
        return torch.exp(-d**2.0 / self.sigma**2.0)
