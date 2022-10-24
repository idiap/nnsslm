#!/usr/bin/env python
"""
train_resnet_2stage.py

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys
import os
import argparse
import math

import random
rng = random.Random()
rng.seed(1920)  # fixed seed

import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import datasets, features, adaptation, utils

import archs

_NEAREST_OUTPUT_DEFAULT = 'nearest'
_ADAPT_METHODS = [_NEAREST_OUTPUT_DEFAULT]


def main(train_path, model_name, extract_ft, output_act, n_out_hidden,
         num_epochs, win_size, hop_size, batch_size, learning_rate, lr_decay,
         pre_model, save_int, sigma, multi_task, all_stype, swf, s1_epoch,
         n_ctx, n_ahd, n_res_blocks, bg_path, mix_dsize, yaxis_symm, cpu,
         adapt_name, adapt_layers, augment_data, idrop_rate, noise_rate,
         noise_lvl, vlvl_ac_std, vlvl_ic_std, loss_buf, num_decay, valid_file,
         grad_clip, n_doa):
    assert not multi_task or not all_stype

    if multi_task:
        prepare_gt = datasets.PrepareGTasSslSns(n_doa=n_doa,
                                                sigma=sigma,
                                                yaxis_symm=yaxis_symm)
    else:
        src_filter = None if all_stype else datasets.src_filter_speech_only
        prepare_gt = datasets.PrepareGTasHeatmap(n_doa=n_doa,
                                                 sigma=sigma,
                                                 yaxis_symm=yaxis_symm,
                                                 src_filter=src_filter)

    # load validation set, if needed
    if valid_file is not None:
        with open(valid_file, 'r') as f:
            valid_filter = utils.CollectionFilter(set(l.strip() for l in f))
        sets = [
            datasets.RawDataset(s,
                                win_size,
                                hop_size,
                                extract_ft=extract_ft,
                                prepare_gt=prepare_gt,
                                sid_filter=valid_filter) for s in train_path
        ]
        valid = datasets.EnsembleDataset(sets)
        train_filter = utils.ReverseFilter(valid_filter)
    else:
        valid = None
        train_filter = None

    # load datasets
    if bg_path is not None:
        # dataset with mixtures on-the-fly
        # mixing configuration
        mconfig = datasets.MixingWithBgConfig([.2, .4, .3, .05, .05], -2, 2, 0,
                                              20, 10)
        # source dataset
        src_sets = datasets.EnsembleDataset([
            datasets.RawDataset(s,
                                win_size,
                                hop_size,
                                active=True,
                                sid_filter=train_filter) for s in train_path
        ])
        # background dataset
        bg_set = datasets.RawDataset(bg_path, win_size, hop_size)

        if idrop_rate > 0.0:
            train = datasets.RandomMixtureWithBgDataset(src_sets,
                                                        bg_set,
                                                        mix_dsize,
                                                        mconfig,
                                                        prepare_gt=prepare_gt,
                                                        reset=True,
                                                        rng=rng)
            train = datasets.DropInputDataset(train,
                                              idrop_rate,
                                              extract_ft=extract_ft,
                                              rng=rng)
        elif noise_rate > 0.0:
            train = datasets.RandomMixtureWithBgDataset(src_sets,
                                                        bg_set,
                                                        mix_dsize,
                                                        mconfig,
                                                        prepare_gt=prepare_gt,
                                                        reset=True,
                                                        rng=rng)
            train = datasets.NoisyInputDataset(train,
                                               noise_rate,
                                               noise_lvl,
                                               extract_ft=extract_ft,
                                               rng=rng)
        elif vlvl_ac_std > 0.0 or vlvl_ic_std > 0.0:
            train = datasets.RandomMixtureWithBgDataset(src_sets,
                                                        bg_set,
                                                        mix_dsize,
                                                        mconfig,
                                                        prepare_gt=prepare_gt,
                                                        reset=True,
                                                        rng=rng)
            train = datasets.VaryingLevelDataset(train,
                                                 vlvl_ac_std,
                                                 vlvl_ic_std,
                                                 extract_ft=extract_ft,
                                                 rng=rng)
        else:
            train = datasets.RandomMixtureWithBgDataset(src_sets,
                                                        bg_set,
                                                        mix_dsize,
                                                        mconfig,
                                                        extract_ft=extract_ft,
                                                        prepare_gt=prepare_gt,
                                                        reset=True,
                                                        rng=rng)
    elif n_ctx > 0 or n_ahd > 0:
        raise NotImplementedError()
        sets = [
            datasets.RawWavContextDataset(s,
                                          extract_ft,
                                          win_size,
                                          hop_size,
                                          win_size // 8 * n_ctx,
                                          win_size // 8 * n_ahd,
                                          prepare_gt=prepare_gt)
            for s in train_path
        ]
        train = datasets.EnsembleDataset(sets)
    else:
        if augment_data is not None and augment_data > 0:
            # sets = [datasets.RawDataset(s, win_size, hop_size) for s in train_path]
            # odata = datasets.EnsembleDataset(sets)
            # adata = datasets.RandomMixtureRealSegmentsDataset(odata, augment_data)
            # sets = [odata, adata]
            raise NotImplementedError()
        if idrop_rate > 0.0:
            sets = [
                datasets.RawDataset(s,
                                    win_size,
                                    hop_size,
                                    sid_filter=train_filter)
                for s in train_path
            ]
            train = datasets.EnsembleDataset(sets, prepare_gt=prepare_gt)
            train = datasets.DropInputDataset(train,
                                              idrop_rate,
                                              extract_ft=extract_ft,
                                              rng=rng)
        elif noise_rate > 0.0:
            sets = [
                datasets.RawDataset(s,
                                    win_size,
                                    hop_size,
                                    sid_filter=train_filter)
                for s in train_path
            ]
            train = datasets.EnsembleDataset(sets, prepare_gt=prepare_gt)
            train = datasets.NoisyInputDataset(train,
                                               noise_rate,
                                               noise_lvl,
                                               extract_ft=extract_ft,
                                               rng=rng)
        elif vlvl_ac_std > 0.0 or vlvl_ic_std > 0.0:
            sets = [
                datasets.RawDataset(s,
                                    win_size,
                                    hop_size,
                                    sid_filter=train_filter)
                for s in train_path
            ]
            train = datasets.EnsembleDataset(sets, prepare_gt=prepare_gt)
            train = datasets.VaryingLevelDataset(train,
                                                 vlvl_ac_std,
                                                 vlvl_ic_std,
                                                 extract_ft=extract_ft,
                                                 rng=rng)
        else:
            sets = [
                datasets.RawDataset(s,
                                    win_size,
                                    hop_size,
                                    extract_ft=extract_ft,
                                    prepare_gt=prepare_gt,
                                    sid_filter=train_filter)
                for s in train_path
            ]
            train = datasets.EnsembleDataset(sets)

    # check input output size
    x0, y0 = train[0]
    input_size = x0.shape

    # init net
    if pre_model is None:
        n_out_map = 2 if multi_task else 1
        net = archs.ResNetTwoStage(input_size,
                                   output_act,
                                   n_out_map=n_out_map,
                                   s2_hidden_size=[500] * n_out_hidden,
                                   n_res_blocks=n_res_blocks,
                                   output_size=n_doa,
                                   roll_padding=not yaxis_symm)
        w_init = archs.WeightInitializer(gain=1.41)
        w_init(net)
    else:
        # load pretrained model
        net = archs.load_module(pre_model)

    if adapt_name is not None:
        assert pre_model is not None
        if adapt_name == _NEAREST_OUTPUT_DEFAULT:
            adapt_func = adaptation.NearestOutputGridSearch(
                prepare_gt, [0, 1, 2])
        else:
            assert False
    else:
        adapt_func = None

    if adapt_layers is not None:
        # train only part of network
        net.set_feat_layers(adapt_layers)
        partial = True
    else:
        partial = False

    criterion = archs.SslSnscLoss(1.0, swf) if multi_task else nn.MSELoss()

    if s1_epoch > 0:
        if n_ctx > 0 or n_ahd > 0:
            nframes = input_size[1]
            s2_crit = archs.Stage1Loss(criterion,
                                       narrow=[(1, n_ctx, nframes - n_ahd)])
        else:
            s2_crit = archs.Stage1Loss(criterion)
        archs.train_stage1(net,
                           model=model_name,
                           dataset=train,
                           num_epochs=s1_epoch,
                           batch_size=batch_size,
                           learning_rate=learning_rate,
                           lr_decay=lr_decay,
                           save_int=save_int,
                           criterion=s2_crit,
                           gpu=not cpu,
                           grad_clip=grad_clip)

    # stoping condition
    if num_decay > 0:
        num_epochs = 0

    archs.train_nn(net,
                   model=model_name,
                   dataset=train,
                   num_epochs=num_epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   lr_decay=lr_decay,
                   save_int=save_int,
                   criterion=criterion,
                   gpu=not cpu,
                   adapt_func=adapt_func,
                   partial=partial,
                   num_lr_decay=num_decay,
                   loss_buf_size=loss_buf,
                   valid_set=valid,
                   grad_clip=grad_clip)


_FEATURES = {'stft': features.FeatureSTFT}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 2-stage ResNet model')
    parser.add_argument('train',
                        metavar='TRAIN_PATH',
                        type=str,
                        nargs='+',
                        help='path to train data and feature, can be '
                        'multiple sets')
    parser.add_argument('-m',
                        '--model',
                        metavar='MODEL_NAME',
                        type=str,
                        default='model',
                        help='model name')
    parser.add_argument('--output-act',
                        type=int,
                        default=0,
                        help='activation at output layer %s' %
                        archs.ACT_INSTRUCTION)
    parser.add_argument('--n-out-hidden',
                        type=int,
                        default=0,
                        help='hidden layers at output phase')
    parser.add_argument('-e',
                        '--epoch',
                        metavar='NEPOCH',
                        type=int,
                        required=True,
                        help='number of epochs')
    parser.add_argument(
        '--s1-epoch',
        metavar='S1_NEPOCH',
        type=int,
        default=0,
        help='(default 0) number of epochs for stage 1 training')
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
    parser.add_argument('--feature',
                        metavar='FEATURE',
                        type=str,
                        default='stft',
                        choices=list(_FEATURES.keys()),
                        help='feature extraction')
    parser.add_argument('--stft-window',
                        metavar='STFT_WIN',
                        type=int,
                        default=2048,
                        help='(default 2048) stft window size')
    parser.add_argument('--stft-hop',
                        metavar='STFT_HOP',
                        type=int,
                        default=1024,
                        help='(default 1024) stft hop size')
    parser.add_argument('-b',
                        '--batch-size',
                        required=True,
                        type=int,
                        metavar='N',
                        help='mini-batch size')
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
    parser.add_argument('--pre-model',
                        metavar='MODEL_NAME',
                        type=str,
                        help='load a pre-trained model')
    parser.add_argument('--save-int',
                        action='store_true',
                        default=False,
                        help='save intermediate models')
    parser.add_argument(
        '--sigma',
        metavar='SIGMA',
        type=float,
        default=5.0,
        help='(default 5.0) ground truth beam width (in degrees)')
    parser.add_argument('--multi-task',
                        action='store_true',
                        help='multi-task: ssl/sns')
    parser.add_argument('--all-stype',
                        action='store_true',
                        default=False,
                        help='consider all sound type as positive samples')
    parser.add_argument('--sns-width-factor',
                        metavar='SWF',
                        type=float,
                        default=1.0,
                        help='(default 1.0) factor to increase sns '
                        'loss beam width (ratio to ssl beam width)')
    parser.add_argument('--context-frames',
                        metavar='N_CTX',
                        type=int,
                        default=0,
                        help='(default 0) number of frames of context')
    parser.add_argument('--ahead-frames',
                        metavar='N_AHD',
                        type=int,
                        default=0,
                        help='(default 0) number of frames to look ahead')
    parser.add_argument('--n-res-blocks',
                        metavar='N_BLKS',
                        type=int,
                        default=5,
                        help='(default 5) number of residual blocks')
    parser.add_argument('--bg-path',
                        metavar='BG_PATH',
                        type=str,
                        help='path to background dataset for mixing')
    parser.add_argument(
        '--mix-dsize',
        metavar='DSIZE',
        type=int,
        default=10000,
        help='(default 10k) number of samples of random mixture'
        ' dataset')
    parser.add_argument('--yaxis-symm',
                        action='store_true',
                        help='linear microphone, symmetric w.r.t.'
                        'the y-axis')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu instead of gpu')
    parser.add_argument('--adaptation',
                        metavar='METHOD',
                        choices=_ADAPT_METHODS,
                        help='Apply an adaptation method instead supervised '
                        'training')
    parser.add_argument('--adapt-layers',
                        metavar='N',
                        type=int,
                        help='(default None) train only the first N layers')
    parser.add_argument(
        '--augment-data',
        metavar='N',
        type=int,
        help='(default None) augment data with number of samples')
    parser.add_argument('--idrop-rate',
                        metavar='RATE',
                        type=float,
                        default=0.0,
                        help='(default 0) rate for drop-out on '
                        'input channels')
    parser.add_argument('--noise-rate',
                        metavar='RATE',
                        type=float,
                        default=0.0,
                        help='(default 0) rate for noisy input')
    parser.add_argument('--noise-lvl',
                        metavar='LVL_DB',
                        type=float,
                        default=-30.0,
                        help='(default -30.0) noise level in dB')
    parser.add_argument('--vlvl-ac-std',
                        metavar='AC_STD',
                        type=float,
                        default=0.0,
                        help='(default 0) standard deviation of '
                        'all channel level change in dB')
    parser.add_argument('--vlvl-ic-std',
                        metavar='IC_STD',
                        type=float,
                        default=0.0,
                        help='(default 0) standard deviation of '
                        'inter-channel level change in dB')
    parser.add_argument('--loss-buf',
                        metavar='BUFFER_SIZE',
                        type=int,
                        default=1000,
                        help='(default 1000) number of batches'
                        ' to calculate loss average')
    parser.add_argument('--num-decay',
                        metavar='NUM_DECAY',
                        type=int,
                        default=0,
                        help='(default 0) if not zero, train until'
                        ' reaching plateau for num_decay times, and in this'
                        ' case number of epochs is ignored')
    parser.add_argument('--valid-file',
                        metavar='VALID_FILE',
                        type=str,
                        default=None,
                        help='file of validation SIDs')
    parser.add_argument('--grad-clip',
                        metavar='MAX_GRAD',
                        type=float,
                        default=None,
                        help='Maximum gradient (abosolute) value '
                        'for gradient clipping')
    parser.add_argument('--n-doa',
                        metavar='N',
                        type=int,
                        default=None,
                        help='(default 360 or 181 depending on yaxis_symm)'
                        ' number of directions')
    args = parser.parse_args()

    # default values
    if args.n_doa is None:
        args.n_doa = 360 if not args.yaxis_symm else 181

    # feature
    if args.feature == 'stft':
        extract_ft = _FEATURES[args.feature](args.stft_window,
                                             args.stft_hop,
                                             min_freq=100,
                                             max_freq=8000)

    main(args.train, args.model, extract_ft, args.output_act,
         args.n_out_hidden, args.epoch, args.window_size, args.hop_size,
         args.batch_size, args.lr, args.ld, args.pre_model, args.save_int,
         args.sigma * math.pi / 180.0, args.multi_task, args.all_stype,
         args.sns_width_factor, args.s1_epoch, args.context_frames,
         args.ahead_frames, args.n_res_blocks, args.bg_path, args.mix_dsize,
         args.yaxis_symm, args.cpu, args.adaptation, args.adapt_layers,
         args.augment_data, args.idrop_rate, args.noise_rate, args.noise_lvl,
         args.vlvl_ac_std, args.vlvl_ic_std, args.loss_buf, args.num_decay,
         args.valid_file, args.grad_clip, args.n_doa)
