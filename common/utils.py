"""
utils.py

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

import numpy as np

import apkit

_DEFAULT_WAV_DIR = 'audio'
_DEFAULT_GTF_DIR = 'gt_frame'
_DEFAULT_GT_DIR  = 'gt_file'
_DEFAULT_BF_DIR  = 'bf'

_WAV_SUFFIX    = '.wav'
_SPK_ID_SUFFIX = '.spkid'

def load_cpsd(afile, win_size, hop_size):
    fs, sig = apkit.load_wav(afile)
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size)
    return apkit.pairwise_cpsd(tf)

def load_gt_cont(path):
    with open(path) as s:
        gt = pickle.load(s)

    ae = np.array([apkit.vec2ae(p) for p, _, _, _, _, _, in gt[4]])
    assert len(ae) == 1
    return ae[0,0]

def load_source_location_utt(path):
    """Load source location(s) at utterance level.

    Args:
        path : path to ground truth label

    Returns:
        gt   : list of ground truth sources' locations
    """
    with open(path) as s:
        gt = pickle.load(s)

    return [p for p, _, _, _, _, _, in gt[4]]

def file2sid(filepath):
    slpos = filepath.rfind('/') + 1
    dotpos = filepath.rfind('.')
    return filepath[slpos:dotpos]

_VAD_RATE = 100

def load_vad_gt(path):
    ph_gt = np.load(path)
    return {k:p != 1 for k,p in ph_gt.items()}

def get_vad_mask_frame(vad, vad_begin, begin, end):
    if end < vad_begin:
        return 0
    iv = max(0, int((begin - vad_begin) * _VAD_RATE))
    jv = min(len(vad), int((end - vad_begin) * _VAD_RATE))
    arate = np.sum(vad[iv:jv]) * 1.0 / _VAD_RATE / (end - begin)
    if arate > .6:
        return 1
    elif arate < .1:
        return 0
    else:
        return None


LOAD_ALL = 0
LOAD_CERTAIN = 1
LOAD_ACTIVE = 2
LOAD_ONE_SOURCE = 3

def load_source_location_frame(root, sid, vad_suffix, load_type,
                               gt_filter=None):
    """Load source location(s) at frame level.

    Args:
        root       : path to database root
        sid        : segment id
        vad_suffix : VAD ground truth file suffix.
        load_type  : {LOAD_ALL: all frames,
                      LOAD_CERTAIN: certain frames (vad == 0 or 1,
                      LOAD_ACTIVE: certain and at least one active source}
        gt_filter  : filter based on ground truth

    Returns:
        gt   : list of (frame_id, gt_locs)
               gt_locs is the list of ground truth sources' locations at
               frame frame_id.
    """
    gtpath = '%s/data/%s.gt.pickle' % (root, sid)
    with open(gtpath) as s:
        gt = pickle.load(s)

    locs = [p for p, _, _, _, _, _, in gt[4]]
    vadpath = '%s/data/%s%s' % (root, sid, vad_suffix)
    vad = np.load(vadpath)

    act = vad == 1.0
    sil = vad == 0.0

    if load_type == LOAD_ALL:
        mask = np.ones(len(vad))
    elif load_type == LOAD_CERTAIN:
        mask = np.all(act + sil, axis=1)
    elif load_type == LOAD_ACTIVE:
        mask = np.all(act + sil, axis=1) * np.any(act, axis=1)
    elif load_type == LOAD_ONE_SOURCE:
        mask = np.all(act + sil, axis=1) * (np.sum(act, axis=1) == 1)
    else:
        assert False

    fids, = mask.nonzero()
    lgt = [(i, [l for j, l in enumerate(locs) if act[i, j]]) for i in fids]
    if gt_filter is not None:
        lgt = [(fid, gt_locs) for fid, gt_locs in lgt
                              if gt_filter(gt_locs)]
    return lgt

_GTF_PATTERN = '%s.w%d_o%d.gtf.pkl'

def load_gtf(path, sid, win_size, hop_size, gtf_dir='data'):
    with open(os.path.join(path, gtf_dir,
                           _GTF_PATTERN % (sid, win_size, hop_size))) as f:
        lgt = pickle.load(f)
    return lgt

def load_gt(path, sid, gt_dir=_DEFAULT_GT_DIR):
    with open(os.path.join(path, gt_dir, '%s.gt.pkl' % sid)) as f:
        lgt = pickle.load(f)
    return lgt

def get_frame_rate(path, wav_dir=_DEFAULT_WAV_DIR):
    sid = next(all_sids(path, wav_dir))
    fs, _, _ = apkit.load_metadata(os.path.join(path, wav_dir, sid + _WAV_SUFFIX))
    return fs

def load_wav(path, sid, wav_dir=_DEFAULT_WAV_DIR):
    return apkit.load_wav(os.path.join(path, wav_dir, sid + _WAV_SUFFIX))

def load_frame(path, sid, fid, win_size, hop_size, wav_dir=_DEFAULT_WAV_DIR):
    return apkit.load_wav(os.path.join(path, wav_dir, sid + _WAV_SUFFIX),
                          offset=fid * hop_size, nsamples=win_size)

_RESULT_DIR = 'results'

def load_pred(path, method, sid, task_id=None):
    rdir = os.path.join(path, _RESULT_DIR, method)
    f = '%s.npy' % sid if task_id is None else '%s_t%d.npy' % (sid, task_id)
    return np.load(os.path.join(rdir, f))

def load_doa(path, method):
    """ DOA index """
    rdir = os.path.join(path, _RESULT_DIR, method)
    return np.load(os.path.join(rdir, 'doas.npy'))

def all_sids(path, wav_dir=_DEFAULT_WAV_DIR):
    """ all segment IDs """
    ddir = os.path.join(path, wav_dir)
    for f in os.listdir(ddir):
        # iterate through all audio files
        if f.endswith(_WAV_SUFFIX):
            # seg id
            yield f[:-len(_WAV_SUFFIX)]

def make_bf_dir(path, bf_name, bf_dir=_DEFAULT_BF_DIR):
    os.makedirs(os.path.join(path, bf_dir, bf_name))

def save_bf_wav(path, bf_name, sid, src_id, fs, sig, spk_id,
                bf_dir=_DEFAULT_BF_DIR):
    prefix = os.path.join(path, bf_dir, bf_name, '%s_%d' % (sid, src_id))
    apkit.save_wav(prefix + _WAV_SUFFIX, fs, sig)
    with open(prefix + _SPK_ID_SUFFIX, 'w') as f:
        print(spk_id, file=f)

class NSrcFilter:
    def __init__(self, nsrc):
        self.nsrc = nsrc

    def __call__(self, gt_locs):
        return len(gt_locs) == self.nsrc

class NSrcNoiseFilter:
    def __init__(self, nsrc, nnoise):
        self.nsrc = nsrc
        self.nnoise = nnoise

    def __call__(self, gt):
        nsrc = 0
        nnoise = 0
        for _, stype, _ in gt:
            if stype == 1:
                nsrc += 1
            else:
                nnoise += 1
        return nsrc == self.nsrc and nnoise == self.nnoise

class NNoiseFilter:
    def __init__(self, nnoise):
        self.nnoise = nnoise

    def __call__(self, gt):
        nnoise = len([1 for _, stype, _ in gt if stype == 0])
        return nnoise == self.nnoise

def SrcNoiseMixFilter(gt):
    nsrc = 0
    nnoise = 0
    for _, stype, _ in gt:
        if stype == 1:
            nsrc += 1
        else:
            nnoise += 1
    return nsrc > 0 and nnoise > 0

class ReverseFilter:
    def __init__(self, filt):
        self.filt = filt

    def __call__(self, gt):
        return not self.filt(gt)

class SrcDistRangeFilter:
    def __init__(self, min_dist, max_dist):
        self.min_dist = min_dist
        self.max_dist = max_dist

    def __call__(self, gt_locs):
        if len(gt_locs) == 2:
            dist = apkit.azimuth_distance(gt_locs[0], gt_locs[1])
            return dist >= self.min_dist and dist < self.max_dist

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

