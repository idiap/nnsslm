#!/usr/bin/env python
"""
extract_cov_mat.py

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

import argparse

import numpy as np

import apkit

_WAVE_SUFFIX = '.wav'
_NPY_SUFFIX = '.npy'

_FREQ_MAX = 8000
_FREQ_MIN = 100

def output_path(wavfile, destdir):
    slash_pos = wavfile.rfind('/') + 1
    return destdir + '/' + wavfile[slash_pos:-len(_WAVE_SUFFIX)] + _NPY_SUFFIX

def main(wavfile, destfile, win_size, hop_size, nfbank, zoom, eps):
    # load signal
    fs, sig = apkit.load_wav(wavfile)
    tf = apkit.stft(sig, apkit.cola_hamming, win_size, hop_size)
    nch, nframe, _ = tf.shape

    # trim freq bins
    nfbin = _FREQ_MAX * win_size / fs            # 0-8kHz
    freq = np.fft.fftfreq(win_size)[:nfbin]
    tf = tf[:,:,:nfbin]

    # compute pairwise gcc on f-banks
    ecov = apkit.empirical_cov_mat(tf, fw=1, tw=1)
    fbw = apkit.mel_freq_fbank_weight(nfbank, freq, fs, fmax=_FREQ_MAX,
                                      fmin=_FREQ_MIN)
    fbcc = apkit.gcc_phat_fbanks(ecov, fbw, zoom, freq, eps=eps)

    # merge to a single numpy array, indexed by 'tpbd'
    #                                           (time, pair, bank, delay)
    feature = np.asarray([fbcc[(i,j)] for i in range(nch)
                                      for j in range(nch)
                                      if i < j])
    feature = np.moveaxis(feature, 2, 0)

    # and map [-1.0, 1.0] to 16-bit integer, to save storage space
    dtype = np.int16
    vmax = np.iinfo(dtype).max
    feature = (feature * vmax).astype(dtype)

    np.save(destfile, feature)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract covariance '
                                     'matrix (upper triangle) as power '
                                     'and phase')
    parser.add_argument('file', metavar='WAV_FILE', type=str,
                        help='path to the wav file')
    parser.add_argument('dest', metavar='DEST_DIR', type=str,
                        help='output directory')
    parser.add_argument('-w', '--window-size', metavar='WIN_SIZE', type=int,
                        default=2048, help='(default 2048) analysis window size')
    parser.add_argument('-o', '--hop-size', metavar='HOP_SIZE', type=int,
                        default=1024, help='(default 1024) hop size, number of samples between windows')
    parser.add_argument('-z', '--cc-zoom', metavar='N', type=int,
                        default=25, help='(default 25) zoom of cross-correlation graph')
    parser.add_argument('-b', '--nfbank', metavar='N', type=int,
                        default=40, help='(default 40) number of filter banks')
    parser.add_argument('-e', '--eps', metavar='EPSILON', type=float,
                        default=0.0, help='(default 0.0) constant added to denominator')
    args = parser.parse_args()
    main(args.file, output_path(args.file, args.dest), args.window_size,
         args.hop_size, args.nfbank, args.cc_zoom, args.eps)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

