"""
testing.py

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import torch
import torch.nn as nn

from .base import SerializableModule, ResidualBlock, _act_funcs, ACT_NONE


class ResNetTwoStageConfig:
    def __init__(self,
                 input_size,
                 output_size=360,
                 output_act=ACT_NONE,
                 n_out_map=1,
                 s1_convs=[],
                 s1_nres=5,
                 s2_convs=[]):
        self.input_size = input_size
        self.output_size = output_size
        self.output_act = output_act
        self.n_out_map = n_out_map
        self.s1_convs = s1_convs
        self.s1_nres = s1_nres
        self.s2_convs = s2_convs


class ResNetTwoStageCustomized(SerializableModule):
    def __init__(self, config):
        super().__init__({'config': config})

        self.config = config

        # feature dimesions
        ic, x, y = config.input_size

        # stage one:
        s1seq = []

        # stage 1 initial convolution (no residual)
        for oc, kernel, stride in config.s1_convs:
            s1seq.append(nn.Conv2d(ic, oc, kernel_size=kernel, stride=stride))
            s1seq.append(nn.BatchNorm2d(oc))
            s1seq.append(nn.ReLU(inplace=True))

            ic = oc
            x = (x - kernel[0] + stride[0]) // stride[0]
            y = (y - kernel[1] + stride[1]) // stride[1]

        # residual layers
        for i in range(config.s1_nres):
            s1seq.append(ResidualBlock(ic, oc))

        # stage one output: map to NDOA directions
        # with multi-task branches
        oc = config.output_size

        s1seq.append(nn.Conv2d(ic, oc * config.n_out_map, kernel_size=1))
        s1seq.append(nn.BatchNorm2d(oc * config.n_out_map))
        s1seq.append(nn.ReLU(inplace=True))

        self.stage1_seq = nn.Sequential(*s1seq)

        # stage two
        s2seq = []

        # swap axes
        # result input size: nfbin, nframe, ndoa
        ic, y = y, oc

        # s2 layers
        for layer, (oc, kernel, stride) in enumerate(config.s2_convs):
            assert stride[1] == 1
            s2seq.append(
                nn.Conv2d(ic * config.n_out_map,
                          oc * config.n_out_map,
                          kernel_size=kernel,
                          stride=stride,
                          groups=config.n_out_map))
            if layer < len(config.s2_convs) - 1:
                s2seq.append(nn.BatchNorm2d(oc * config.n_out_map))
                s2seq.append(nn.ReLU(inplace=True))
            elif config.output_act != ACT_NONE:
                s2seq.append(_act_funcs[config.output_act]())

            ic = oc
            x = (x - kernel[0] + stride[0]) // stride[0]
            # y = y                 # rolling pading in doa axes

        # check output size
        # output size: (1, 1, output_size) * n_out_map
        assert oc == 1
        assert x == 1
        assert y == config.output_size

        self.stage2 = nn.Sequential(*s2seq)

    def stage1(self, x):
        # input  : ndata, nch, nframe, nfbin
        x = self.stage1_seq(x)
        # now    : ndata, ndoa * n_out_map, nframe, nfbin

        dims = x.size()
        assert dims[1] == self.config.output_size * self.config.n_out_map

        x = x.view(dims[0], self.config.output_size, self.config.n_out_map,
                   dims[2], dims[3])
        # now    : ndata, ndoa, n_out_map, nframe, nfbin

        # output : ndata, nfbin, nframe, n_out_map, ndoa
        return x.permute(0, 4, 3, 2, 1)

    def forward(self, x):
        # input  : ndata, nch, nframe, nfbin
        ndata = x.size(0)

        # stage one:
        x = self.stage1(x)
        # now    : ndata, nfbin, nframe, n_out_map, ndoa

        # stage two:
        x = x.permute(0, 1, 3, 2, 4)
        # now    : ndata, nfbin, n_out_map, nframe, ndoa

        dims = x.size()
        x = x.contiguous().view(dims[0], dims[1] * dims[2], dims[3], dims[4])
        # now    : ndata, nfbin * n_out_map, nframe, ndoa

        # rolling padding in doa axes
        pad_size = sum(
            [kernel[1] - 1 for _, kernel, _ in self.config.s2_convs])
        x = torch.cat((x, x.narrow(3, 0, pad_size)), dim=3)

        # convs
        output = self.stage2(x)

        assert tuple(output.size()) == \
                (ndata, self.config.n_out_map, 1, self.config.output_size)

        # output : ndata, n_out_hidden, ndoa
        return output.squeeze()


class FullyConvMaxPoolOut(SerializableModule):
    def __init__(self, input_size, convs, output_act=ACT_NONE, drop_out_p=0.0):
        super().__init__({
            'input_size': input_size,
            'convs': convs,
            'output_act': output_act,
            'drop_out_p': drop_out_p
        })

        # feature dimesions
        ic, x, y = input_size

        # sequence
        seq = []

        # convolutions
        for lid, (oc, kernel, stride) in enumerate(convs):
            seq.append(nn.Conv2d(ic, oc, kernel_size=kernel, stride=stride))
            if lid < len(convs) - 1:
                seq.append(nn.BatchNorm2d(oc))
                seq.append(nn.ReLU(inplace=True))
            elif output_act != ACT_NONE:
                seq.append(_act_funcs[output_act]())

            ic = oc
            x = (x - kernel[0] + stride[0]) // stride[0]
            y = (y - kernel[1] + stride[1]) // stride[1]

        # maxpool
        seq.append(nn.MaxPool2d((x, 1), stride=1))

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.seq(x)
        return x.squeeze()


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
