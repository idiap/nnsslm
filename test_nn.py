#!/usr/bin/env python
"""
test_nn.py

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

sys.path.append('..')
import common

import archs

_FEAT_SUFFIX = '.npy'

def _predict(net, x):
    batch = 256
    result = []
    for i in xrange(0, len(x), batch):
        j = min(i + batch, len(x))
        y = Variable(torch.from_numpy(x[i:j])).cuda()
        outputs = net(y)
        result.append(outputs.cpu().data.numpy())
    return np.concatenate(result)

def main(test_path, model, fname, method, add_sns):
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
    fdir = os.path.join(test_path, 'features', fname)
    for f in os.listdir(fdir):
        if f.endswith(_FEAT_SUFFIX):
            name = f[:-len(_FEAT_SUFFIX)]
            print >> sys.stderr, name
            feat = np.load(os.path.join(fdir, f))
            odtype = feat.dtype
            feat = feat.astype('float32', copy=False)
            if np.issubdtype(odtype, np.integer):
                feat /= abs(float(np.iinfo(odtype).min)) #normalize

            # prediction
            pred = _predict(net, feat)
            np.save(os.path.join(rdir, name), np.moveaxis(pred, -1, 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test multi-source NN model')
    parser.add_argument('test', metavar='TEST_PATH', type=str,
                        help='path to test data and feature')
    parser.add_argument('model', metavar='MODEL_PATH', type=str,
                        help='path to trained model')
    parser.add_argument('-n', '--fname', metavar='FEATURE', type=str,
                        required=True, help='feature name')
    parser.add_argument('-m', '--method', metavar='METHOD', type=str,
                        required=True, help='method name')
    parser.add_argument('--add-sns', action='store_true',
                        help='add constant sns to output')
    args = parser.parse_args()
    main(args.test, args.model, args.fname, args.method, args.add_sns)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

