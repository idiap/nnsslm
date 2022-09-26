"""
datasets.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>

This file is part of "Neural Network based Sound Source Localization Models".

"Neural Network based Sound Source Localization Models" is free software:
you can redistribute it and/or modify it under the terms of the BSD 3-Clause
License.

"Neural Network based Sound Source Localization Models" is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the BSD 3-Clause License for more details.
"""

import os
import pickle as pickle
import random
import math

import numpy as np
import scipy.special
import torch.utils.data as data

import apkit

import utils

_DATA_DIR = 'data'
_FEATURE_DIR = 'features'
_BEAMFORMING_DIR = 'bf'

_GT_SUFFIX = '.gt.pickle'
_GTF_SUFFIX_PATTERN = '.w%d_o%d.gtf.pkl'
_WAV_SUFFIX = '.wav'
_FEATURE_SUFFIX = '.npy'
_BF_DATA_SUFFIX = '.npy'
_STYPE_GT_SUFFIX = '.stype.pkl'
_MDIST_SUFFIX = '.mdist'
_AF_PATTERN = '%s.w%d_o%d.af.npy'

_N_CLASSES = 360
_AZI_DIRECTIONS = np.arange(_N_CLASSES, dtype='float32') \
                                        * 2 * np.pi / _N_CLASSES

# binaural mixture
_MAX_SIG_LVL   =   5.0
_MIN_SIG_LVL   = -20.0
_MAX_NOISE_LVL =   5.0
_MIN_NOISE_LVL = -40.0
_MAX_SNR       =  20.0
_MIN_SNR       =   0.0
_MAX_SIG_DIF   =  10.0
assert(_MAX_SIG_DIF < _MAX_SNR - _MIN_SNR)

# TDOA ground truth
# -1ms to 1ms, step size 0.01ms, half time step if fs=48kHz
_TDOA_GT_VALUES  = np.linspace(-1e-3, 1e-3, 201)
_TDOA_GT_SIGMA   = 50e-6    # 50 micro sec

# other constants
_SOUND_SPEED = 343.0    # m/s

# sound types:
_STYPE_SPEECH     = 1
_STYPE_NON_SPEECH = 0

def prepare_gt_as_class(gt_locs):
    assert len(gt_locs) <= 1
    if len(gt_locs) == 1:
        azi, ele = apkit.vec2ae(gt_locs[0])
        return int((azi + np.pi) // (2 * np.pi / _N_CLASSES))
    else:
        return 0        # assume one and only one source

_SIGMA = np.pi * 5 / 180    # 5 degrees

def prepare_gt_as_heatmap(gt_locs):
    hmap = np.zeros(_N_CLASSES)
    aindex = np.arange(_N_CLASSES, dtype='float32') \
                                        * 2 * np.pi / _N_CLASSES
    for l in gt_locs:
        azi, ele = apkit.vec2ae(l)
        d = np.remainder(aindex - azi, 2 * np.pi)
        d[d > np.pi] -= 2 * np.pi
        h = np.exp(-d ** 2 / _SIGMA ** 2)
        hmap = np.maximum(hmap, h)
    return hmap.astype('float32', copy=False)


def _gt_azimuth_distance(src_doa, ref_doas):
    azi, ele = apkit.vec2ae(src_doa)
    d = np.remainder(ref_doas - azi, 2 * np.pi)
    d[d > np.pi] -= 2 * np.pi
    return d

def _gt_yaxis_symm_distance(src_doa, ref_doas):
    ysym_azi = apkit.vec2ysyma(src_doa)
    d = ref_doas - ysym_azi
    return d

class PrepareGTasHeatmap(object):
    def __init__(self, n_doa=_N_CLASSES, sigma=np.pi * 5 / 180,
                 yaxis_symm=False, src_filter=None,
                 dtype='float32'):
        """
            yaxis_symm :
                If the microphone array is linear (e.g. kinect),
                the localization is sysmmetric w.r.t. an axis.
                Here, we assume it is w.r.t the y-axis.
        """
        self.sigma      = sigma
        self.src_filter = src_filter
        self.dtype      = dtype

        if not yaxis_symm:
            self.aindex = np.arange(n_doa, dtype=dtype) \
                                                * 2 * np.pi / n_doa
            self.dmetric = _gt_azimuth_distance
        else:
            self.aindex = np.linspace(-np.pi / 2, np.pi / 2, n_doa,
                                      dtype=dtype)
            self.dmetric = _gt_yaxis_symm_distance

    def __call__(self, gt_srcs):
        res = np.zeros(len(self.aindex), dtype=self.dtype)

        if self.src_filter is not None:
            gt_srcs = [src for src in gt_srcs if self.src_filter(src)]
        if len(gt_srcs) > 0:
            ctb = np.zeros((len(gt_srcs), res.shape[0]),
                           dtype=self.dtype)
            for i, src in enumerate(gt_srcs):
                loc, stype, spk_id = src
                d = self.dmetric(loc, self.aindex)
                ctb[i] = np.exp(-d ** 2 / self.sigma ** 2)

            # localization map (likelihood)
            res = np.max(ctb, axis=0)

        return res

class PrepareGTasSslSns(PrepareGTasHeatmap):
    """ multi-task (ssl + sns) ground truth """

    def __init__(self, n_doa=_N_CLASSES, sigma=np.pi * 5 / 180,
                 yaxis_symm=False, src_filter=None,
                 dtype='float32'):
        super(PrepareGTasSslSns, self).__init__(n_doa, sigma,
                                                yaxis_symm,
                                                src_filter, dtype)

    def __call__(self, gt_srcs):
        res = np.zeros((2, len(self.aindex)), dtype=self.dtype)

        if self.src_filter is not None:
            gt_srcs = [src for src in gt_srcs if self.src_filter(src)]
        if len(gt_srcs) > 0:
            ctb = np.zeros((len(gt_srcs), res.shape[1]))
            for i, src in enumerate(gt_srcs):
                loc, stype, spk_id = src
                d = self.dmetric(loc, self.aindex)
                ctb[i] = np.exp(-d ** 2 / self.sigma ** 2)
            stype = np.array([t for _, t, _ in gt_srcs],
                             dtype=self.dtype)

            # localization map (likelihood)
            res[0] = np.max(ctb, axis=0)

            # speech/non-speech map
            res[1] = stype[np.argmax(ctb, axis=0)]
        return res

def src_filter_speech_only(src):
    loc, stype, spk_id = src
    return stype == _STYPE_SPEECH

def prepare_gt_as_posterior(gt_locs):
    post = np.zeros(_N_CLASSES)
    for l in gt_locs:
        azi, ele = apkit.vec2ae(l)
        post[int(azi / (2 * np.pi / _N_CLASSES) + 0.5) % _N_CLASSES] = 1.0
    return post.astype('float32', copy=False)

_OPOST_N_DIR = 72

def prepare_gt_as_ordered_posterior(gt_locs):
    assert len(gt_locs) <= 2
    labels = []
    for l in gt_locs:
        azi, ele = apkit.vec2ae(l)
        labels.append(int(azi / (2 * np.pi / _OPOST_N_DIR) + 0.5) % _OPOST_N_DIR)
    for _ in range(2 - len(labels)):
        labels.append(_OPOST_N_DIR)
    labels = sorted(labels)
    post = np.zeros(2 * (_OPOST_N_DIR + 1))
    post[labels[0]] = 1.0
    post[labels[1] + _OPOST_N_DIR + 1] = 1.0
    return post.astype('float32', copy=False)

def prepare_gtf_location_as_heatmap(gt):
    gt_locs = [loc for loc, _, _ in gt]
    return prepare_gt_as_heatmap(gt_locs)

def prepare_gtf_speech_as_heatmap(gt):
    gt_locs = [loc for loc, stype, _ in gt if stype == 1]
    return prepare_gt_as_heatmap(gt_locs)

def prepare_gtf_location_and_snsc(gt):
    res = np.zeros((2, _N_CLASSES), dtype='float32')

    if len(gt) > 0:
        ctb = np.zeros((len(gt), _N_CLASSES))
        for i, (l, _, _) in enumerate(gt):
            azi, ele = apkit.vec2ae(l)
            d = np.remainder(_AZI_DIRECTIONS - azi, 2 * np.pi)
            d[d > np.pi] -= 2 * np.pi
            ctb[i] = np.exp(-d ** 2 / _SIGMA ** 2)
        stype = np.array([t for _, t, _ in gt], dtype='float32')

        # localization map (likelihood)
        res[0] = np.max(ctb, axis=0)

        # speech/non-speech map
        res[1] = stype[np.argmax(ctb, axis=0)]

    return res

def prepare_gt_as_tdoa(mdist, doas):
    tdoas = mdist * np.cos(np.asarray(doas)) / _SOUND_SPEED
    nsrc = len(tdoas)
    if nsrc > 0:
        spec = np.max(np.exp(-(np.repeat(_TDOA_GT_VALUES, nsrc).reshape((-1,nsrc))
                                - tdoas) ** 2 / _TDOA_GT_SIGMA ** 2), axis=1)
    else:
        spec = np.zeros(len(_TDOA_GT_VALUES))
    return spec.astype(np.float32, copy=False)

def get_hmap_doas(n_doa=_N_CLASSES):
    aindex = np.arange(n_doa, dtype='float32') * 2 * np.pi / n_doa
    return np.array([np.cos(aindex), np.sin(aindex), np.zeros(n_doa)]).T

def get_opost_doas():
    aindex = np.arange(_OPOST_N_DIR, dtype='float32') \
                                        * 2 * np.pi / _OPOST_N_DIR
    return np.array([np.cos(aindex), np.sin(aindex), np.zeros(_OPOST_N_DIR)]).T

def prepare_gt_none(gt_locs):
    return 0

def load_feat(path):
    return np.load(path).astype('float32', copy=False)

class SslDataset(data.Dataset):
    """SSL dataset
    """
    def __init__(self, path, feature_name, prepare_gt=prepare_gt_as_class):
        """
        Args:
            path         : path to the dataset.
            feature_name : name of feature
        """
        self.datadir = os.path.join(path, _DATA_DIR)
        self.featdir = os.path.join(path, _FEATURE_DIR, feature_name)
        self.prepare_gt = prepare_gt
        assert os.path.isdir(self.datadir)
        assert os.path.isdir(self.featdir)

        self.names = [f[:-len(_GT_SUFFIX)]
                        for f in os.listdir(self.datadir)
                        if f.endswith(_GT_SUFFIX)]

    def __getitem__(self, index):
        n = self.names[index]
        feat = load_feat(os.path.join(self.featdir, n + _FEATURE_SUFFIX))
        gt_locs = utils.load_source_location_utt(
                            os.path.join(self.datadir, n + _GT_SUFFIX))
        gt = self.prepare_gt(gt_locs)
        return (feat, gt)

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

class FrameDataset(data.Dataset):
    """Dataset at frame level
    """
    LOAD_ALL = utils.LOAD_ALL
    LOAD_CERTAIN = utils.LOAD_CERTAIN
    LOAD_ACTIVE = utils.LOAD_ACTIVE

    def __init__(self, path, feature_name, vad_suffix,
                 load_type=LOAD_ACTIVE, prepare_gt=prepare_gt_as_class):
        """
        Args:
            path       : path to the dataset.
            feature_name : name of feature
            vad_suffix : VAD ground truth file suffix.
            non_sil    : non-silent frames only.
        """
        datadir = os.path.join(path, _DATA_DIR)
        self.featdir = os.path.join(path, _FEATURE_DIR, feature_name)
        self.prepare_gt = prepare_gt
        assert os.path.isdir(datadir)
        assert os.path.isdir(self.featdir)

        self.names = []
        self.fids = []
        self.gts = []

        for f in os.listdir(datadir):
            if f.endswith(_GT_SUFFIX):
                n = f[:-len(_GT_SUFFIX)]
                flocs = utils.load_source_location_frame(path, n, vad_suffix, load_type)
                for fid, gt_locs in flocs:
                    self.names.append(n)
                    self.fids.append(fid)
                    self.gts.append(gt_locs)

    def __getitem__(self, index):
        n = self.names[index]
        fid = self.fids[index]
        data = np.load(os.path.join(self.featdir, n + _FEATURE_SUFFIX),
                       mmap_mode='r')
        feat = np.asarray(data[fid])
        odtype = feat.dtype
        feat = feat.astype('float32', copy=False)
        if np.issubdtype(odtype, np.integer):
            feat /= abs(float(np.iinfo(odtype).min)) #normalize
        gt = self.prepare_gt(self.gts[index])
        return (feat, gt)

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

class SingleNumpyDataset(data.Dataset):
    def __init__(self, path):
        self.data = np.load(path)
        self.names = [path for _ in range(len(self.data))]

    def __getitem__(self, index):
        feat = self.data[index]
        odtype = feat.dtype
        feat = feat.astype('float32', copy=False)
        if np.issubdtype(odtype, np.integer):
            feat /= abs(float(np.iinfo(odtype).min)) #normalize
        return (feat, 0)

    def __len__(self):
        return len(self.data)

    def get_names(self):
        return self.names

class EnsembleDataset(data.Dataset):
    """Ensemble of list of datasets.
    """
    def __init__(self, datasets, extract_ft=None, prepare_gt=None):
        """
        Args:
            datasets : list of datasets
            extract_ft : function to extract feature
            prepare_gt : function to prepare ground truth
        """
        self.datasets = datasets
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt
        self.index = [(i, j) for i, d in enumerate(datasets) for j in range(len(d))]

    def __getitem__(self, index):
        i, j = self.index[index]
        x, y = self.datasets[i][j]
        if self.extract_ft is not None:
            x = self.extract_ft(*x)
        if self.prepare_gt is not None:
            y = self.prepare_gt(y)
        return x, y

    def __len__(self):
        return len(self.index)

class AverageDataset(data.Dataset):
    """Dataset by averaging (over time) another dataset
    """
    def __init__(self, orig, alpha):
        """
        Args:
            orig  : original dataset
            alpha : weighting factor (see Xiao et. al.)
        """
        self.names = orig.get_names()
        self.items = [(self.weighted_mean(feat, alpha), gt)
                                               for feat, gt in orig]

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

    def weighted_mean(self, feat, alpha):
        nc, nt, nf = feat.shape
        weights = np.sum(np.abs(feat) ** alpha, axis=2).repeat(nf).reshape((feat.shape))
        return np.average(feat, axis=1, weights=weights)

class RawWavDataset(data.Dataset):
    """Dataset from raw wav data and then apply feature extraction

    @deprecated, use RawDataset instead
    """

    def __init__(self, path, extract_ft, win_size, hop_size, prepare_gt=None):
        """
        Args:
            path       : path to the dataset.
            extract_ft : function to extract feature
            win_size   : window size
            hop_size   : hop size
            prepare_gt : function to prepare ground truth
        """
        self.datadir = os.path.join(path, _DATA_DIR)
        assert os.path.isdir(self.datadir)

        self.win_size = win_size
        self.hop_size = hop_size
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt

        self.names = []
        self.fids = []
        self.gts = []

        for f in os.listdir(self.datadir):
            if f.endswith(_WAV_SUFFIX):
                n = f[:-len(_WAV_SUFFIX)]

                if prepare_gt is not None:
                    flocs = utils.load_gtf(path, n, win_size, hop_size)
                    for fid, gt in flocs:
                        self.names.append(n)
                        self.fids.append(fid)
                        self.gts.append(gt)
                else:
                    _, _, nsamples = apkit.load_metadata(
                                                    os.path.join(self.datadir, f))
                    nframes = (nsamples - win_size) // hop_size
                    self.names += [n] * nframes
                    self.fids += list(range(nframes))

    def __getitem__(self, index):
        n = self.names[index]
        fid = self.fids[index]
        fs, sig = apkit.load_wav(os.path.join(self.datadir, n + _WAV_SUFFIX),
                                 offset=fid * self.hop_size,
                                 nsamples=self.win_size)
        assert sig.shape[1] == self.win_size
        feat = self.extract_ft(fs, sig)
        if self.prepare_gt is not None:
            gt = self.prepare_gt(self.gts[index])
        else:
            gt = None
        return (feat, gt)

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

class RawWavContextDataset(data.Dataset):
    """Dataset from raw wav data with context and then apply feature extraction
    """

    def __init__(self, path, extract_ft, win_size, hop_size, ctx_size,
                 ahd_size, prepare_gt=prepare_gt_none):
        """
        Args:
            path       : path to the dataset.
            extract_ft : function to extract feature
            win_size   : window size
            hop_size   : hop size
            ctx_size   : context size in the past
            ahd_size   : context size to look ahead
            prepare_gt : function to prepare ground truth
        """
        self.datadir = os.path.join(path, _DATA_DIR)
        assert os.path.isdir(self.datadir)

        self.win_size = win_size
        self.hop_size = hop_size
        self.ctx_size = ctx_size
        self.ahd_size = ahd_size
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt

        self.names = []
        self.fids = []
        self.gts = []

        for f in os.listdir(self.datadir):
            if f.endswith(_WAV_SUFFIX):
                n = f[:-len(_WAV_SUFFIX)]
                flocs = utils.load_gtf(path, n, win_size, hop_size)
                for fid, gt in flocs:
                    self.names.append(n)
                    self.fids.append(fid)
                    self.gts.append(gt)

    def __getitem__(self, index):
        n = self.names[index]
        fid = self.fids[index]
        f_start = fid * self.hop_size
        c_start = max(0, f_start - self.ctx_size)
        c_end = f_start + self.win_size + self.ahd_size
        fs, sig = apkit.load_wav(os.path.join(self.datadir, n + _WAV_SUFFIX),
                                 offset=c_start, nsamples=(c_end - c_start))
        n_ctx_pad = self.ctx_size - (f_start - c_start)
        n_ahd_pad = (c_end - c_start) - sig.shape[1]
        if n_ctx_pad > 0 or n_ahd_pad > 0:
            sig = np.concatenate((np.zeros((sig.shape[0], n_ctx_pad)),
                                  sig,
                                  np.zeros((sig.shape[0], n_ahd_pad))),
                                 axis=1)
        assert sig.shape[1] == self.win_size + self.ctx_size + self.ahd_size
        feat = self.extract_ft(fs, sig)
        odtype = feat.dtype
        feat = feat.astype('float32', copy=False)
        if np.issubdtype(odtype, np.integer):
            feat /= abs(float(np.iinfo(odtype).min)) #normalize
        gt = self.prepare_gt(self.gts[index])
        return (feat, gt)

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

class PreCompDataset(data.Dataset):
    """Dataset with precomputed features
    """

    def __init__(self, path, feature_name, win_size, hop_size,
                 prepare_gt=prepare_gt_as_class):
        """
        Args:
            path       : path to the dataset.
            feature_name : name of feature
            win_size   : window size
            hop_size   : hop size
            prepare_gt : function to prepare ground truth
        """
        datadir = os.path.join(path, _DATA_DIR)
        self.featdir = os.path.join(path, _FEATURE_DIR, feature_name)
        self.prepare_gt = prepare_gt
        assert os.path.isdir(datadir)
        assert os.path.isdir(self.featdir)

        self.names = []
        self.fids = []
        self.gts = []

        for f in os.listdir(datadir):
            if f.endswith(_WAV_SUFFIX):
                n = f[:-len(_WAV_SUFFIX)]
                flocs = utils.load_gtf(path, n, win_size, hop_size)
                for fid, gt_locs in flocs:
                    self.names.append(n)
                    self.fids.append(fid)
                    self.gts.append(gt_locs)

    def __getitem__(self, index):
        n = self.names[index]
        fid = self.fids[index]
        data = np.load(os.path.join(self.featdir, n + _FEATURE_SUFFIX),
                       mmap_mode='r')
        feat = np.asarray(data[fid])
        odtype = feat.dtype
        feat = feat.astype('float32', copy=False)
        if np.issubdtype(odtype, np.integer):
            feat /= abs(float(np.iinfo(odtype).min)) #normalize
        gt = self.prepare_gt(self.gts[index])
        return (feat, gt)

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

class BfDataset(data.Dataset):
    """Beamforming Dataset ?
    """

    def __init__(self, path, bf_name, prepare_gt=(lambda x: x)):
        """
        Args:
            path       : path to the dataset.
            feature_name : name of feature
            win_size   : window size
            hop_size   : hop size
            prepare_gt : function to prepare ground truth
        """
        self.datadir = os.path.join(path, _BEAMFORMING_DIR, bf_name)
        self.prepare_gt = prepare_gt
        assert os.path.isdir(self.datadir)

        self.names = []
        self.fids = []
        self.gts = []

        for f in os.listdir(self.datadir):
            if f.endswith(_STYPE_GT_SUFFIX):
                n = f[:-len(_STYPE_GT_SUFFIX)]
                with open(os.path.join(self.datadir, f)) as s:
                    stypes = pickle.load(s)
                self.names += [n] * len(stypes)
                self.fids += range(len(stypes))
                self.gts += stypes

    def __getitem__(self, index):
        n = self.names[index]
        fid = self.fids[index]
        data = np.load(os.path.join(self.datadir, n + _BF_DATA_SUFFIX),
                       mmap_mode='r')
        feat = np.asarray(data[fid])
        odtype = feat.dtype
        feat = feat.astype('float32', copy=False)
        if np.issubdtype(odtype, np.integer):
            feat /= abs(float(np.iinfo(odtype).min)) #normalize
        gt = self.prepare_gt(self.gts[index])
        return (feat, gt)

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

class BinauralSingleDatatset:
    def __init__(self, path, win_size, hop_size, doa_gt=False, vad_gt=False):
        """
        Args:
            path     : path to the dataset.
            win_size : window size
            hop_size : hop size
            doa_gt   : (default False) if DOA ground truth is available
            vad_gt   : (default False) if VAD ground truth is available
        """
        self.datadir = os.path.join(path, _DATA_DIR)
        assert os.path.isdir(self.datadir)

        self.win_size = win_size
        self.hop_size = hop_size

        self.sids = {}
        self.fids = {}
        self.doas = {}

        for f in os.listdir(self.datadir):
            if f.endswith(_WAV_SUFFIX):
                sid = f[:-len(_WAV_SUFFIX)]
                self._load_data(sid, doa_gt, vad_gt)

    def get_random_frame(self, mdist, rng=random):
        """
        Args:
            mdist : distance between microphones
            rng   : python random number generator (random.Random)

        Returns:
            doa       : DOA (right is zero, left is pi) if any, otherwise None
            (fs, sig) : time domain signal, indexed by "ct"
        """
        sid = rng.choice(self.sids[self._make_index(mdist)])
        active_frames = self.fids[sid]
        if type(active_frames) is list:
            fid = rng.choice(active_frames)
        else:
            assert type(active_frames) is int
            fid = rng.randrange(active_frames)
        return self.doas[sid], self._load_frame(sid, fid)

    def _load_data(self, sid, doa_gt, vad_gt):
        if doa_gt:
            with open(os.path.join(self.datadir, sid + _GT_SUFFIX)) as s:
                _, _, mdist, doa, _, _, _, _, _ = pickle.load(s)
        else:
            with open(os.path.join(self.datadir, sid + _MDIST_SUFFIX)) as s:
                mdist = float(next(s).strip())
                doa = None

        if vad_gt:
            aframes = list(np.load(os.path.join(self.datadir, _AF_PATTERN \
                                        % (sid, self.win_size, self.hop_size))))
        else:
            fs, _, nsamples = apkit.load_metadata(os.path.join(self.datadir,
                                                               sid + _WAV_SUFFIX))
            aframes = (nsamples - self.win_size) // self.hop_size

        if self._make_index(mdist) in self.sids:
            l = self.sids[self._make_index(mdist)]
        else:
            l = []
            self.sids[self._make_index(mdist)] = l
        l.append(sid)

        self.doas[sid] = doa
        self.fids[sid] = aframes

    def _make_index(self, mdist):
        return int(round(mdist * 1000))

    def _load_frame(self, sid, fid):
        return apkit.load_wav(os.path.join(self.datadir, sid + _WAV_SUFFIX),
                              offset=fid * self.hop_size,
                              nsamples=self.win_size)

class BinauralMixtureDatatset(data.Dataset):
    """Randomly generate mixtures from single binauaral data
    """
    def __init__(self, signal_set, noise_set, size, mdists, nsrc_dist,
                 extract_ft, prepare_gt=prepare_gt_as_tdoa, reset=True,
                 rng=random):
        """
        Args:
            signal_set : BinauralSingleDatatset of signal
            noise_set  : BinauralSingleDatatset of noise
            size       : total number frames in the dataset
            mdists     : different distances between microphones
            nsrc_dist  : list of floats, distribution of number of sources
            extract_ft : function to extract feature
            prepare_gt : function to prepare ground truth
            reset      : reset random number generator when before reading the
                         first frame
            rng        : python random number generator (random.Random)
        """
        self.signal_set = signal_set
        self.noise_set  = noise_set
        self.mdists     = mdists
        self.size       = size

        nsrc_dist = np.asarray(nsrc_dist)
        nsrc_dist /= np.sum(nsrc_dist)
        self.nsrc_dist  = nsrc_dist

        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt
        self.init_state = rng.getstate() if reset else None
        self.rng = rng

    def __getitem__(self, index):
        if index == 0 and self.init_state is not None:
            self.rng.setstate(self.init_state)

        mdist = self.mdists[index % len(self.mdists)]
        noise_lvl, signal_lvls = self._get_noise_src_level()

        _, (fs, frame) = self.noise_set.get_random_frame(mdist, rng=self.rng)
        frame *= np.power(10.0, noise_lvl * 0.05)

        doas = []
        for signal_lvl in signal_lvls:
            doa, (nfs, sig) = self.signal_set.get_random_frame(mdist,
                                                               rng=self.rng)
            assert nfs == fs
            doas.append(doa)
            frame += sig * np.power(10.0, signal_lvl * 0.05)

        return self.extract_ft(fs, frame), self.prepare_gt(mdist, doas)

    def __len__(self):
        return self.size

    def _get_noise_src_level(self):
        rn = self.rng.random()
        nsrc = 0
        s = 0.0
        for p in self.nsrc_dist:
            s += p
            if rn < s:
                break;
            else:
                nsrc += 1

        if nsrc == 0:
            return self.rng.uniform(_MIN_NOISE_LVL, _MAX_NOISE_LVL), []
        else:
            max_l = _MIN_SIG_LVL
            min_l = _MAX_SIG_LVL
            sig_lvls = []
            for _ in range(nsrc):
                if len(sig_lvls) == 0:
                    slvl = self.rng.uniform(_MIN_SIG_LVL, _MAX_SIG_LVL)
                else:
                    slvl = self.rng.uniform(
                                max(_MIN_SIG_LVL, max_l - _MAX_SIG_DIF),
                                min(_MAX_SIG_LVL, min_l + _MAX_SIG_DIF))
                sig_lvls.append(slvl)
                if slvl > max_l:
                    max_l = slvl
                if slvl < min_l:
                    min_l = slvl
            assert max_l - min_l <= _MAX_SIG_DIF
            nlvl = self.rng.uniform(max_l - _MAX_SNR, min_l - _MIN_SNR)
            return nlvl, sig_lvls

class MixtureDataset(data.Dataset):
    """Dataset of mixture of clean speech with silent (ego-noise) recordings
    """

    def __init__(self, signal_set, noise_set, extract_ft,
                 sig_vol_range=(0.25, 1.0), rng=random):
        self.signal_set = signal_set
        self.noise_set  = noise_set
        self.extract_ft = extract_ft
        logvola, logvolb = np.log(sig_vol_range)
        self.vols = [math.exp(rng.uniform(logvola, logvolb))
                                            for _ in range(len(signal_set))]

    def __getitem__(self, index):
        (fs, frame), gt = self.signal_set[index]
        (nfs, nframe), _ = self.noise_set[index % len(self.noise_set)]
        assert nfs == fs
        frame = frame * self.vols[index] + nframe
        return self.extract_ft(fs, frame), gt

    def __len__(self):
        return len(self.signal_set)

class RawDataset(data.Dataset):
    def __init__(self, path, win_size, hop_size, active=False,
                 extract_ft=(lambda x, y: (x, y)), prepare_gt=(lambda x: x),
                 wav_dir=_DATA_DIR, gtf_dir=_DATA_DIR):
        """
        Args:
            path     : path to the dataset.
            win_size : window size
            hop_size : hop size
            active   : load active frames only
            wav_dir  : directory to wav files (relative to `path')
            gtf_dir  : directory to ground truth files (relative to `path')
        """
        self.gtf_dir = os.path.join(path, gtf_dir)
        self.wav_dir = os.path.join(path, wav_dir)
        assert os.path.isdir(self.gtf_dir)
        assert os.path.isdir(self.wav_dir)

        self.win_size = win_size
        self.hop_size = hop_size
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt

        self.frames = []

        gt_suffix = _GTF_SUFFIX_PATTERN % (win_size, hop_size)

        for f in os.listdir(self.gtf_dir):
            if f.endswith(gt_suffix):
                sid = f[:-len(gt_suffix)]
                self._load_gt(sid, f, active)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        sid, fid, src_gt = self.frames[index]
#       print >> sys.stderr, 'load (%s, %d)' % (sid, fid)
        fs, frame = self._load_frame(sid, fid)
        return self.extract_ft(fs, frame), self.prepare_gt(src_gt)

    def _load_gt(self, sid, gt_file, active):
        with open(os.path.join(self.gtf_dir, gt_file)) as s:
            gtf = pickle.load(s)

        for fid, srcs in gtf:
            if len(srcs) > 0 or not active:
                self.frames.append((sid, fid, srcs))

    def _load_frame(self, sid, fid):
        return apkit.load_wav(os.path.join(self.wav_dir, sid + _WAV_SUFFIX),
                              offset=fid * self.hop_size,
                              nsamples=self.win_size)

class MixingWithBgConfig():
    def __init__(self, nsrc_dist, min_bg_fluc, max_bg_fluc, min_snr,
                 max_snr, max_sig_dif):
        """
        Args:
            nsrc_dist  : list of floats, distribution of number of sources
        """
        assert(max_sig_dif < max_snr - min_snr)

        nsrc_dist = np.asarray(nsrc_dist)
        nsrc_dist /= np.sum(nsrc_dist)
        self.nsrc_dist  = nsrc_dist

        self.min_bg_fluc = min_bg_fluc
        self.max_bg_fluc = max_bg_fluc
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.max_sig_dif = max_sig_dif

class RandomMixtureWithBgDataset(data.Dataset):
    """Randomly generate mixtures from single-source data
    """
    def __init__(self, signal_set, bg_set, size, config,
                 extract_ft=(lambda x, y: (x, y)),
                 prepare_gt=(lambda x: x), reset=True, rng=random):
        """
        Args:
            signal_set : dataset of signal
            bg_set     : dataset of bg
            size       : total number frames in the dataset
            config     : mixing config, see MixingConfig
            extract_ft : function to extract feature
            prepare_gt : function to prepare ground truth
            reset      : reset random number generator when before reading the
                         first frame
            rng        : python random number generator (random.Random)
        """
        self.signal_set = signal_set
        self.bg_set     = bg_set
        self.size       = size
        self.config     = config
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt
        self.init_state = rng.getstate() if reset else None
        self.rng = rng

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError
        if index == 0 and self.init_state is not None:
            self.rng.setstate(self.init_state)

        bg_fluc, signal_snrs = self._get_bg_src_level()

        (fs, frame), _ = self.rng.choice(self.bg_set)
        frame *= np.power(10.0, bg_fluc / 20.0)
        noise_lvl = apkit.power_avg_db(frame)

        sig_datas = self.rng.sample(self.signal_set, len(signal_snrs))
        src_gt = []
        for signal_snr, sig_data in zip(signal_snrs, sig_datas):
            ((sig_fs, sig_frame), sig_gt) = sig_data
            assert sig_fs == fs
            src_gt += sig_gt
            signal_lvl = noise_lvl + signal_snr
            frame += sig_frame * np.power(10.0, (signal_lvl -
                                            apkit.power_avg_db(sig_frame)) / 20.0)

        return self.extract_ft(fs, frame), self.prepare_gt(src_gt)

    def __len__(self):
        return self.size

    def _get_bg_src_level(self):
        rn = self.rng.random()
        nsrc = 0
        s = 0.0
        for p in self.config.nsrc_dist:
            s += p
            if rn < s:
                break;
            else:
                nsrc += 1

        bg_fluc = self.rng.uniform(self.config.min_bg_fluc,
                                   self.config.max_bg_fluc)
        if nsrc == 0:
            return bg_fluc, []
        else:
            max_snr = self.config.min_snr
            min_snr = self.config.max_snr
            sig_snrs = []
            for _ in range(nsrc):
                if len(sig_snrs) == 0:
                    ssnr = self.rng.uniform(self.config.min_snr,
                                            self.config.max_snr)
                else:
                    ssnr = self.rng.uniform(
                                max(self.config.min_snr, max_snr - self.config.max_sig_dif),
                                min(self.config.max_snr, min_snr + self.config.max_sig_dif))
                sig_snrs.append(ssnr)
                if ssnr > max_snr:
                    max_snr = ssnr
                if ssnr < min_snr:
                    min_snr = ssnr
            assert max_snr - min_snr <= self.config.max_sig_dif
            return bg_fluc, sig_snrs

def prepare_gt_as_nsrc(gt_srcs):
    return len(gt_srcs)

class RandomMixtureRealSegmentsDataset(data.Dataset):
    """
    Randomly generate mixtures from single-source data.
    The single source data have background noise.
    The mixture keeps the background noise in the same level.
    """
    def __init__(self, signal_set, size, max_sig_dif=1.0, nsrc=2, act_snr=0.0,
                 extract_ft=(lambda x, y: (x, y)),
                 prepare_gt=(lambda x: x), reset=True, rng=random,
                 single_out=None):
        """
        Args:
            signal_set  : dataset of signal
            size        : total number frames in the dataset
            max_sig_dif : maximum difference between sources snr
            nsrc        : nubmer of sources to mix
            act_snr     : snr threshold, the segments with lower snr are not used
            extract_ft  : function to extract feature
            prepare_gt  : function to prepare ground truth
            reset       : reset random number generator when before reading the
                          first frame
            rng         : python random number generator (random.Random)
            single_out  : if not None, use only one single segments
                          that with index single_out
        """
        self.signal_set  = signal_set
        self.size        = size
        self.max_sig_dif = max_sig_dif
        self.nsrc        = nsrc
        self.extract_ft  = extract_ft
        self.prepare_gt  = prepare_gt
        self.init_state  = rng.getstate() if reset else None
        self.rng         = rng
        self.single_out  = single_out

        # measure backgroud power
        bg_vols = []
        for (fs, x), y in signal_set:
            if len(y) == 0:
                bg_vols.append(apkit.power_avg(x))
        bg_vol = np.mean(bg_vols)
        vthres = bg_vol * (10.0 ** (act_snr / 10.0) + 1.0)

        # indexing
        self.list = []
        for i, ((fs, x), y) in enumerate(signal_set):
            if len(y) == 1:
                v = apkit.power_avg(x)
                if v >= vthres:
                    self.list.append((i, v - bg_vol))

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError
        if index == 0 and self.init_state is not None:
            self.rng.setstate(self.init_state)

        sl = self.rng.sample(self.list, self.nsrc)
        xys = [self.signal_set[i] for i, v in sl]
        amps = self._get_amps([v for i, v in sl])

        # print volume for analysis
        print('$volume$ %d %s %s' % (self.nsrc,
                                     ' '.join(['%.6g' % v for i, v in sl]),
                                     ' '.join(['%.6g' % a for a in amps])))
        ###########################

        if self.single_out is None:
            frame = np.sum([x * a for ((fs, x), y), a
                                          in zip(xys, amps)], axis=0)
            gt = [g for x, y in xys for g in y]
        else:
            frame = xys[self.single_out][0][1]
            gt = xys[self.single_out][1]
        fs = xys[0][0][0]

        return self.extract_ft(fs, frame), self.prepare_gt(gt)

    def _get_amps(self, vols):
        # target db : beta ditribution
        tardb = [scipy.special.btdtri(3, 3, self.rng.random())
                                            for _ in range(self.nsrc)]
        tardb = np.array(tardb) * 10.0 - 5.0

        # target volume
        tarvol = 10.0 ** (tardb / 10.0)

        # amplification in volume (power)
        ramps = tarvol / np.asarray(vols)
        # keep constant background noise
        ramps = ramps / np.sum(ramps)

        # amplification in amplitude
        return np.sqrt(ramps)

    def __len__(self):
        return self.size

def store_dataset(dataset, prefix):
    (fs, sig), _ = dataset[0]
    nch, win_size = sig.shape

    # split in groups
    ngroup = 1000
    wpt = prefix + '-%06d' + _WAV_SUFFIX
    gpt = prefix + '-%06d' + _GTF_SUFFIX_PATTERN % (win_size, win_size / 2)
    for i in range(0, len(dataset), ngroup):
        gc = i // ngroup
        print('g#%03d' % gc)
        j = min(len(dataset), i + ngroup)

        lsig = []
        lgt = []
        for k in range(i, j):
            (nfs, sig), gt = dataset[k]
            assert nfs == fs
            assert sig.shape == (nch, win_size)
            lsig.append(sig)
            lgt.append(gt)

        # save wav
        sig = np.concatenate(lsig, axis=1)
        apkit.save_wav(wpt % gc, fs, sig)

        # save gtf
        gtf = [(2 * f, g) for f, g in enumerate(lgt)]
        with open(gpt % gc, 'w') as f:
            pickle.dump(gtf, f)

class RawDecomposedDataset(data.Dataset):
    def __init__(self, path, win_size, hop_size, active=False,
                 extract_ft=(lambda x, y: (x, y)),
                 prepare_gt=(lambda x: x)):
        """
        Args:
            path     : path to the dataset.
            win_size : window size
            hop_size : hop size
            active   : load active frames only
        """
        self.mixed = RawDataset(path, win_size, hop_size, active, extract_ft,
                                prepare_gt)
        self.single1 = RawDataset(path + '_s0', win_size, hop_size, active,
                                  extract_ft, prepare_gt)
        self.single2 = RawDataset(path + '_s1', win_size, hop_size, active,
                                  extract_ft, prepare_gt)

        assert len(self.mixed) == len(self.single1)
        assert len(self.mixed) == len(self.single2)

    def __len__(self):
        return len(self.mixed)

    def __getitem__(self, index):
        x, y = self.mixed[index]
        x1, y1 = self.single1[index]
        x2, y2 = self.single2[index]
        return np.stack([x, x1, x2]), np.stack([y, y1, y2])

class DropInputDataset(data.Dataset):
    def __init__(self, orig_set, drop_rate, rng=random, extract_ft=None,
                 prepare_gt=None):
        self.orig_set = orig_set
        self.drop_rate = drop_rate
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt
        self.rng = random

    def __getitem__(self, index):
        (fs, sig), y = self.orig_set[index]

        if self.rng.random() < self.drop_rate:
            sig[self.rng.randint(0, len(sig) - 1)] = 0
        x = (fs, sig)

        if self.extract_ft is not None:
            x = self.extract_ft(*x)
        if self.prepare_gt is not None:
            y = self.prepare_gt(y)
        return x, y

    def __len__(self):
        return len(self.orig_set)

class NoisyInputDataset(data.Dataset):
    def __init__(self, orig_set, noise_rate, noise_lvl, rng=random,
                 extract_ft=None, prepare_gt=None):
        self.orig_set = orig_set
        self.noise_rate = noise_rate
        self.noise_lvl = noise_lvl
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt
        self.rng = random

    def __getitem__(self, index):
        (fs, sig), y = self.orig_set[index]

        if self.rng.random() < self.noise_rate:
            noise = 10.0 ** (self.noise_lvl / 20.0) \
                                * np.random.randn(*sig.shape[1:])
            sig[self.rng.randint(0, len(sig) - 1)] += noise
        x = (fs, sig)

        if self.extract_ft is not None:
            x = self.extract_ft(*x)
        if self.prepare_gt is not None:
            y = self.prepare_gt(y)
        return x, y

    def __len__(self):
        return len(self.orig_set)

class VaryingLevelDataset(data.Dataset):
    """ add noise by varying input signal volume"""
    def __init__(self, orig_set, ac_var, ic_var, rng=random,
                 extract_ft=None, prepare_gt=None):
        """
            ac_var : variance of all channel level change in dB
            ic_var : variance of inter-channel level change in dB
        """
        self.orig_set = orig_set
        self.ac_var = ac_var
        self.ic_var = ic_var
        self.extract_ft = extract_ft
        self.prepare_gt = prepare_gt
        self.rng = random

    def __getitem__(self, index):
        (fs, sig), y = self.orig_set[index]

        # shift all channels
        ac_shift = self.rng.gauss(0, self.ac_var)

        # shift differently for each channel
        for i in range(len(sig)):
            ic_shift = self.rng.gauss(0, self.ic_var)
            sig[i] *= 10.0 ** ((ac_shift + ic_shift) / 20.0)

        x = (fs, sig)

        if self.extract_ft is not None:
            x = self.extract_ft(*x)
        if self.prepare_gt is not None:
            y = self.prepare_gt(y)
        return x, y

    def __len__(self):
        return len(self.orig_set)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

