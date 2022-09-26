"""
archs.py

Network architectures

Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import itertools

import torch
import torch.nn as nn

from .base import SerializableModule, _act_funcs, ACT_NONE, ACT_SIGMOID


class MLP(SerializableModule):
    def __init__(self,
                 layer_size,
                 hidden_act=ACT_SIGMOID,
                 output_act=ACT_NONE,
                 batch_norm=False):
        super().__init__({
            'layer_size': layer_size,
            'hidden_act': hidden_act,
            'output_act': output_act,
            'batch_norm': batch_norm,
        })

        nlayer = len(layer_size) - 1
        seq = []
        for i in range(nlayer):
            seq.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < nlayer - 1:
                seq.append(_act_funcs[hidden_act]())
                if batch_norm:
                    seq.append(nn.BatchNorm1d(layer_size[i + 1]))
            elif output_act != ACT_NONE:
                seq.append(_act_funcs[output_act]())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x.view(x.size(0), -1))


def _cnn_submodule(in_channels, out_channels, batch_norm):
    seq = [
        nn.Conv2d(in_channels, out_channels, 5, 2, 2),
        nn.ReLU(inplace=True)
    ]
    if batch_norm:
        seq.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*seq)


class CNN(SerializableModule):
    def __init__(self,
                 layer_nch,
                 input_size,
                 output_act=ACT_NONE,
                 batch_norm=False):
        super().__init__({
            'layer_nch': layer_nch,
            'input_size': input_size,
            'output_act': output_act,
            'batch_norm': batch_norm
        })
        self.mseq = nn.Sequential()

        _, x, y = input_size
        for i in range(len(layer_nch) - 1):
            self.mseq.add_module(
                'sub%d' % i,
                _cnn_submodule(layer_nch[i], layer_nch[i + 1], batch_norm))
            x = (x + 1) // 2
            y = (y + 1) // 2

        # output sequence
        outseq = [nn.Linear(layer_nch[-1] * x * y, 360)]
        if output_act != ACT_NONE:
            outseq.append(_act_funcs[output_act]())
        self.out = nn.Sequential(*outseq)

    def forward(self, x):
        x = self.mseq(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class RegionFC(SerializableModule):
    def __init__(self,
                 input_size,
                 s1_fb_nsize=5,
                 s1_hidden_size=[],
                 s2_azi_nsize=11,
                 s2_hidden_size=[],
                 output_act=ACT_NONE,
                 batch_norm=False):
        super().__init__({
            'input_size': input_size,
            's1_fb_nsize': s1_fb_nsize,
            's1_hidden_size': s1_hidden_size,
            's2_azi_nsize': s2_azi_nsize,
            's2_hidden_size': s2_hidden_size,
            'output_act': output_act,
            'batch_norm': batch_norm
        })
        npair, nfbank, ndelay = input_size
        self.s1_fb_nsize = s1_fb_nsize
        self.s2_azi_nsize = s2_azi_nsize

        # stage one: npair * fb_nsize * ndelay -> hidden layers -> 360
        s1seq = []
        hsizes = itertools.chain(s1_hidden_size, [360])
        # first layer conv kernel across all delay
        osize = next(hsizes)
        s1seq.append(nn.Conv2d(npair, osize,
                               kernel_size=(s1_fb_nsize, ndelay)))
        s1seq.append(nn.ReLU(inplace=True))
        if batch_norm:
            s1seq.append(nn.BatchNorm2d(osize))
        isize = osize
        # rest layers 1 by 1 kernel
        for osize in hsizes:
            s1seq.append(nn.Conv2d(isize, osize, kernel_size=1))
            s1seq.append(nn.ReLU(inplace=True))
            if batch_norm:
                s1seq.append(nn.BatchNorm2d(osize))
            isize = osize
        self.stage1 = nn.Sequential(*s1seq)

        # stage two: s1o_nfb * azi_nsize -> hidden layers -> 1
        # input should be arange as:
        #           (1, 360 + azi_nszie - 1, fbank - fb_nsize + 1)
        s2seq = []
        hsizes = itertools.chain(s2_hidden_size, [1])
        # first layer conv kernel across all fbank
        osize = next(hsizes)
        s2seq.append(
            nn.Conv2d(1,
                      osize,
                      kernel_size=(s2_azi_nsize, nfbank - s1_fb_nsize + 1)))
        isize = osize
        # rest layers 1 by 1 kernel
        for osize in hsizes:
            s2seq.append(nn.ReLU(inplace=True))
            if batch_norm:
                s2seq.append(nn.BatchNorm2d(isize))
            s2seq.append(nn.Conv2d(isize, osize, kernel_size=1))
            isize = osize
        if output_act != ACT_NONE:
            s2seq.append(_act_funcs[output_act]())
        self.stage2 = nn.Sequential(*s2seq)

    def forward(self, x):
        ndata, npair, nfbank, ndelay = x.size()
        s1o = self.stage1(x)
        # s1 output size: (360, nfbank-fb_nsize+1, 1)
        # swap axes + padding
        s2i = s1o.permute(0, 3, 1, 2)
        s2i = torch.cat((s2i, s2i.narrow(2, 0, self.s2_azi_nsize - 1)), dim=2)
        # s2 input size: (1,360+azi_nsize-1,nfbank-fb_nsize+1)
        s2o = self.stage2(s2i).view(ndata, -1)
        return s2o


##############################################################################
######## Legacy problem, I have to import the classes in this module  ########
######## so that I can unpickle the class            o(>_<)o          ########
##############################################################################
from .multitask import ResNetTwoStage
from .obsolete import ResNetCtx32, ResNetClassification, ResNet, ResNetv2
##############################################################################

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
