#!/usr/bin/env python
"""
test_nn_raw.py

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

import sys
import os
import argparse

import numpy as np
import torch
from torch.autograd import Variable

import apkit

sys.path.append('..')
import common

import archs

_WAV_SUFFIX = '.wav'

def _predict(net, x, batch=100):
    result = []
    for i in xrange(0, len(x), batch):
        j = min(i + batch, len(x))
        y = Variable(torch.from_numpy(x[i:j])).cuda()
        outputs = net(y)
        result.append(outputs.cpu().data.numpy())
    return np.concatenate(result)

def _pad_context(sig, offset, size, ctx_size, ahd_size):
    nch, nsamples = sig.shape
    c_start = max(0, offset - ctx_size)
    n_ctx_pad = c_start - (offset - ctx_size)
    c_end = min(nsamples, offset + size + ahd_size)
    n_ahd_pad = (offset + size + ahd_size) - c_end
    if n_ctx_pad > 0 or n_ahd_pad > 0:
        res = np.concatenate((np.zeros((nch, n_ctx_pad)),
                              sig[:,c_start:c_end],
                              np.zeros((nch, n_ahd_pad))),
                             axis=1)
    else:
        res = sig[:,c_start:c_end]
    return res

def _load_feature(datafile, extract_ft, win_size, hop_size, n_ctx, n_ahd):
    fs, sig = apkit.load_wav(datafile)
    nch, nsamples = sig.shape
    feat = np.array([extract_ft(fs, _pad_context(sig, o, win_size,
                                                 n_ctx * win_size / 8,
                                                 n_ahd * win_size / 8))
                     for o in range(0, nsamples - win_size + 1, hop_size)])
    return feat

def main(test_path, model, extract_ft, win_size, hop_size, n_ctx, n_ahd, method,
         add_sns, batch):
    # init net
    net = archs.load_module(model)
    if add_sns:
        net = archs.AddConstantSns(net)
    print >> sys.stderr, net
    net.eval()
    net.cuda()

    # create result folder
    rdir = os.path.join(test_path, 'results', method)
    os.makedirs(rdir)

    # load and save doas
    doa = common.get_hmap_doas()
    np.save(os.path.join(rdir, 'doas'), doa)

    # iterate through all data
    ddir = os.path.join(test_path, 'data')
    for f in os.listdir(ddir):
        if f.endswith(_WAV_SUFFIX):
            name = f[:-len(_WAV_SUFFIX)]
            print >> sys.stderr, name
            feat = _load_feature(os.path.join(ddir, f), extract_ft, win_size,
                                 hop_size, n_ctx, n_ahd)
            odtype = feat.dtype
            feat = feat.astype('float32', copy=False)
            if np.issubdtype(odtype, np.integer):
                feat /= abs(float(np.iinfo(odtype).min)) #normalize

            # prediction
            pred = _predict(net, feat, batch)
            np.save(os.path.join(rdir, name), np.moveaxis(pred, -1, 0))

_FEATURES = {'stft' : common.FeatureSTFT}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test multi-source NN model '
                                                 'frow raw wav data')
    parser.add_argument('test', metavar='TEST_PATH', type=str,
                        help='path to test data and feature')
    parser.add_argument('model', metavar='MODEL_PATH', type=str,
                        help='path to trained model')
    parser.add_argument('-n', '--feature', metavar='FEATURE', type=str,
                        required=True, choices=_FEATURES.keys(),
                        help='feature extraction method')
    parser.add_argument('-m', '--method', metavar='METHOD', type=str,
                        required=True, help='method name')
    parser.add_argument('-w', '--window-size', metavar='WIN_SIZE',
                        type=int, default=2048,
                        help='(default 2048) analysis window size')
    parser.add_argument('-o', '--hop-size', metavar='HOP_SIZE', type=int,
                        default=1024,
                        help='(default 1024) hop size, number of samples between windows')
    parser.add_argument('--wframes-per-block', metavar='N_WFRAME', type=int,
                        default=4, help='(default 4) number of whole frames in on block')
    parser.add_argument('--context-frames', metavar='N_CTX', type=int,
                        default=0, help='number of frames of context')
    parser.add_argument('--ahead-frames', metavar='N_AHD', type=int,
                        default=0, help='number of frames to look ahead')
    parser.add_argument('--add-sns', action='store_true',
                        help='add constant sns to output')
    parser.add_argument('--batch-size', metavar='BATCH', type=int,
                        default=100, help='size of a batch')
    args = parser.parse_args()

    if args.feature == 'stft':
        frame_size = args.window_size / args.wframes_per_block
        extract_ft = _FEATURES[args.feature](frame_size, frame_size / 2,
                                             min_freq=100, max_freq=8000)

    main(args.test, args.model, extract_ft, args.window_size, args.hop_size,
         args.context_frames, args.ahead_frames, args.method, args.add_sns,
         args.batch_size)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

