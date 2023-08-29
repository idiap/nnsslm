#!/usr/bin/env python
"""
train_rfc_ts.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys
import argparse

sys.path.append('..')
import common

import archs


def main(train_path, fname, model, s1_fb_nsize, s1_hidden_size, s2_azi_nsize,
         s2_hidden_size, batch_norm, output_act, num_epochs, win_size,
         hop_size, batch_size, learning_rate, lr_decay, resume, save_int,
         init_sigma, end_sigma, s1_epoch):
    # load dataset
    sets = [
        common.PreCompDataset(s,
                              fname,
                              win_size,
                              hop_size,
                              prepare_gt=common.prepare_gtf_speech_as_heatmap)
        for s in train_path
    ]
    train = common.EnsembleDataset(sets)

    # check input output size
    x0, y0 = train[0]
    input_size = x0.shape

    # init net
    if not resume:
        net = archs.RegionFC(input_size,
                             s1_fb_nsize,
                             s1_hidden_size,
                             s2_azi_nsize,
                             s2_hidden_size,
                             output_act=output_act,
                             batch_norm=batch_norm)
    else:
        # load saved model
        net = archs.load_module(model)

    archs.train_stage1(net,
                       model=model,
                       dataset=train,
                       num_epochs=s1_epoch,
                       batch_size=batch_size,
                       learning_rate=learning_rate,
                       lr_decay=lr_decay,
                       save_int=save_int,
                       init_sigma=init_sigma,
                       end_sigma=end_sigma)

    archs.train_nn(net,
                   model=model,
                   dataset=train,
                   num_epochs=num_epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   lr_decay=lr_decay,
                   save_int=save_int,
                   init_sigma=init_sigma,
                   end_sigma=end_sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-source local receptive field (2 stage)')
    parser.add_argument(
        'train',
        metavar='TRAIN_PATH',
        type=str,
        nargs='+',
        help='path to train data and feature, can be multiple sets')
    parser.add_argument('-n',
                        '--fname',
                        metavar='FEATURE',
                        type=str,
                        required=True,
                        help='feature name')
    parser.add_argument('-m',
                        '--model',
                        metavar='MODEL_NAME',
                        type=str,
                        default='model',
                        help='model name')
    parser.add_argument('--s1-nsize',
                        default=5,
                        type=int,
                        help='neighbor size of stage one')
    parser.add_argument('--s1-hsize',
                        default=[],
                        type=int,
                        metavar='N',
                        nargs='+',
                        help='number of units of stage one hidden layer')
    parser.add_argument('--s2-nsize',
                        default=11,
                        type=int,
                        help='neighbor size of stage two')
    parser.add_argument('--s2-hsize',
                        default=[],
                        type=int,
                        metavar='N',
                        nargs='+',
                        help='number of units of stage two hidden layer')
    parser.add_argument('--output-act',
                        type=int,
                        default=0,
                        help='activation at output layer %s' %
                        archs.ACT_INSTRUCTION)
    parser.add_argument('--batch-norm',
                        action='store_true',
                        default=False,
                        help='batch normalization at all except output layers')
    parser.add_argument('-e',
                        '--epoch',
                        metavar='NEPOCH',
                        type=int,
                        default=100,
                        help='number of epochs')
    parser.add_argument('--s1-epoch',
                        metavar='S1_NEPOCH',
                        type=int,
                        default=10,
                        help='number of epochs for stage 1 training')
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
    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        metavar='LR',
                        type=float,
                        default=0.001,
                        help='initial learning rate')
    parser.add_argument('--ld',
                        '--lr-decay',
                        metavar='DECAY',
                        type=int,
                        default=10,
                        help='number of epochs to halve learning rate')
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='resume training, load model with the same name')
    parser.add_argument('--save-int',
                        action='store_true',
                        default=False,
                        help='save intermediate models')
    parser.add_argument('--init-sigma',
                        metavar='SIGMA',
                        type=float,
                        default=5.0,
                        help='(default 5.0) initial ground '
                        'truth beam width (in degrees)')
    parser.add_argument('--end-sigma',
                        metavar='SIGMA',
                        type=float,
                        default=5.0,
                        help='(default 5.0) end (of first '
                        'epoch) ground truth beam width (in degrees)')
    args = parser.parse_args()
    main(args.train, args.fname, args.model, args.s1_nsize, args.s1_hsize,
         args.s2_nsize, args.s2_hsize, args.batch_norm, args.output_act,
         args.epoch, args.window_size, args.hop_size, args.batch_size, args.lr,
         args.ld, args.resume, args.save_int, args.init_sigma, args.end_sigma,
         args.s1_epoch)

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
