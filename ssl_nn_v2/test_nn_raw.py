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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common import utils, doa, features, inference, misc

import archs


def main(test_path, model, extract_ft, win_size, hop_size, n_ctx, n_ahd,
         method, add_sns, batch, yaxis_symm, gpu, forward_kargs):
    # init net
    net = archs.load_module(model)
    if add_sns:
        net = archs.AddConstantSns(net)
    print(net, file=sys.stderr)
    net.eval()
    if gpu:
        net.cuda()

    # create result folder
    rdir = os.path.join(test_path, 'results', method)
    os.makedirs(rdir)

    # number of directions
    n_doa = None

    # iterate through all data
    for sid in utils.all_sids(test_path):
        print(sid, file=sys.stderr)
        fs, sig = utils.load_wav(test_path, sid)
        feat = inference.extract_framewise_feature(fs,
                                                   sig,
                                                   win_size,
                                                   hop_size,
                                                   extract_ft,
                                                   ctx_size=n_ctx,
                                                   ahd_size=n_ahd,
                                                   dtype='float32')
        # prediction
        pred = inference.predict_batch(net, feat, batch, gpu, forward_kargs)
        assert n_doa is None or n_doa == pred.shape[-1]
        n_doa = pred.shape[-1]
        np.save(os.path.join(rdir, sid), np.moveaxis(pred, -1, 0))

    # load and save doas
    doas = doa.sample_azimuth_3d(n_doa, yaxis_symm)
    np.save(os.path.join(rdir, 'doas'), doas)


_FEATURES = {'stft': features.FeatureSTFT}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test multi-source NN model '
                                     'frow raw wav data')
    parser.add_argument('test',
                        metavar='TEST_PATH',
                        type=str,
                        help='path to test data and feature')
    parser.add_argument('model',
                        metavar='MODEL_PATH',
                        type=str,
                        help='path to trained model')
    parser.add_argument('-n',
                        '--feature',
                        metavar='FEATURE',
                        type=str,
                        required=True,
                        choices=list(_FEATURES.keys()),
                        help='feature extraction method')
    parser.add_argument('-m',
                        '--method',
                        metavar='METHOD',
                        type=str,
                        required=True,
                        help='method name')
    parser.add_argument('-w',
                        '--window-size',
                        metavar='WIN_SIZE',
                        type=int,
                        default=2048,
                        help='(default 2048) analysis window size')
    parser.add_argument(
        '-o',
        '--hop-size',
        metavar='HOP_SIZE',
        type=int,
        default=1024,
        help='(default 1024) hop size, number of samples between windows')
    parser.add_argument('--wframes-per-block',
                        metavar='N_WFRAME',
                        type=int,
                        default=4,
                        help='(default 4) number of whole frames in on block')
    parser.add_argument('--context-frames',
                        metavar='N_CTX',
                        type=int,
                        default=0,
                        help='number of frames of context')
    parser.add_argument('--ahead-frames',
                        metavar='N_AHD',
                        type=int,
                        default=0,
                        help='number of frames to look ahead')
    parser.add_argument('--add-sns',
                        action='store_true',
                        help='add constant sns to output')
    parser.add_argument('--batch-size',
                        metavar='BATCH',
                        type=int,
                        default=100,
                        help='size of a batch')
    parser.add_argument('--yaxis-symm',
                        action='store_true',
                        help='linear microphone, symmetric w.r.t. the y-axis')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu instead of gpu')
    parser.add_argument('--forward-kargs',
                        type=misc.parse_dict,
                        default={},
                        help='model forward function keyword arguments '
                        '(example `k1:v1,k2:v2`')
    args = parser.parse_args()

    if args.feature == 'stft':
        frame_size = args.window_size // args.wframes_per_block
        extract_ft = _FEATURES[args.feature](frame_size,
                                             frame_size // 2,
                                             min_freq=100,
                                             max_freq=8000)

    main(args.test, args.model, extract_ft, args.window_size, args.hop_size,
         args.context_frames, args.ahead_frames, args.method, args.add_sns,
         args.batch_size, args.yaxis_symm, not args.cpu, args.forward_kargs)
