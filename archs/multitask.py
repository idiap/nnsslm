"""
multitask.py

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import itertools

import torch
import torch.nn as nn

from .base import (SerializableModule, ResidualBlock, _act_funcs, ACT_NONE,
                   ACT_SIGMOID)
from .gradcam import GradCamable


class MultiTaskMLP(nn.Module):
    def __init__(self,
                 input_size,
                 structures,
                 hidden_act=ACT_SIGMOID,
                 output_act=ACT_NONE,
                 batch_norm=False):
        super().__init__()

        self.subnets = nn.ModuleList()
        for layer_size in structures:
            seq = []
            isize = input_size
            for osize in layer_size[:-1]:
                seq.append(nn.Linear(isize, osize))
                if batch_norm:
                    seq.append(nn.BatchNorm1d(osize))
                seq.append(_act_funcs[hidden_act]())
                isize = osize
            osize = layer_size[-1]
            seq.append(nn.Linear(isize, osize))
            if output_act != ACT_NONE:
                seq.append(_act_funcs[output_act]())
            self.subnets.append(nn.Sequential(*seq))

    def forward(self, x):
        outs = [sub.forward(x) for sub in self.subnets]
        return torch.cat(outs, dim=1)


class ResNetTwoStage(SerializableModule, GradCamable):
    def __init__(self,
                 input_size,
                 output_act=ACT_NONE,
                 n_out_map=1,
                 s2_hidden_size=[],
                 s2_azi_nsize=5,
                 output_size=360,
                 n_res_blocks=5,
                 roll_padding=True):
        super().__init__({
            'input_size': input_size,
            'output_act': output_act,
            'n_out_map': n_out_map,
            's2_hidden_size': s2_hidden_size,
            's2_azi_nsize': s2_azi_nsize,
            'output_size': output_size,
            'n_res_blocks': n_res_blocks,
            'roll_padding': roll_padding
        })

        self.output_size = output_size
        self.s2_azi_nsize = s2_azi_nsize
        self.roll_padding = roll_padding
        self.feat_layers = 1

        ic, x, y = input_size

        # stage one:
        s1seq = []

        # initial layers (no residual)
        # layer 1
        oc = 4 * ic
        s1seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 7), stride=(1, 3)))
        s1seq.append(nn.BatchNorm2d(oc))
        s1seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 7 + 3) // 3

        # layer 2
        oc = 4 * ic
        s1seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 5), stride=(1, 2)))
        s1seq.append(nn.BatchNorm2d(oc))
        s1seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 5 + 2) // 2

        # residual layers
        for _ in range(n_res_blocks):
            s1seq.append(ResidualBlock(ic, oc))

        # stage one trunk
        self.stage1trunk = nn.Sequential(*s1seq)

        # stage one output: map to 360 directions
        # output size should depend on roll_padding
        if roll_padding:
            s1_output_size = output_size
        else:
            s1_output_size = output_size + s2_azi_nsize - 1
        self.stage1out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ic, s1_output_size, kernel_size=1),
                          nn.BatchNorm2d(s1_output_size),
                          nn.ReLU(inplace=True)) for _ in range(n_out_map)
        ])

        # stage two
        s2seqs = []
        for _ in range(n_out_map):
            s2seq = []

            # input size: nfbin, nframe, ndoa
            ic = y

            # hidden layers
            for oc in s2_hidden_size:
                s2seq.append(nn.Conv2d(ic, oc, kernel_size=1))
                s2seq.append(nn.BatchNorm2d(oc))
                s2seq.append(nn.ReLU(inplace=True))
                ic = oc

            # output layer
            # output size: 1, 1, 360
            oc = 1
            s2seq.append(nn.Conv2d(ic, oc, kernel_size=(x, s2_azi_nsize)))
            s2seq.append(_act_funcs[output_act]())

            s2seqs.append(s2seq)
        self.stage2 = nn.ModuleList([nn.Sequential(*s) for s in s2seqs])

    def _stage1_internal(self, x):
        # input  : ndata, nch, nframe, nfbin
        x = self.stage1trunk(x)
        x = torch.stack([branch(x) for branch in self.stage1out])
        # now    : n_out_map, ndata, ndoa, nframe, nfbin
        # output : ndata, nfbin, nframe, n_out_map, ndoa
        return x.permute(1, 4, 3, 0, 2)

    def stage1(self, x):
        s1_internal = self._stage1_internal(x)
        if self.roll_padding:
            return s1_internal
        else:
            # stage 1 output consider padding when roll_padding=False
            return s1_internal.narrow(4, self.s2_azi_nsize // 2,
                                      self.output_size)

    def forward(self, x, no_out_act=False):
        # input  : ndata, nch, nframe, nfbin
        x = self._stage1_internal(x)
        # now    : ndata, nfbin, nframe, n_out_hidden, ndoa
        return self._stage2_internal(x, no_out_act)

    def _stage2_internal(self, x, no_out_act=False):
        # padding
        if self.roll_padding:
            x = torch.cat((x, x.narrow(4, 0, self.s2_azi_nsize - 1)), dim=4)
        # stage two:
        if not no_out_act:
            output = torch.stack([
                branch(x[:, :, :, bid, :])
                for bid, branch in enumerate(self.stage2)
            ],
                                 dim=3)
        else:
            output = torch.stack([
                branch[:-1](x[:, :, :, bid, :])
                for bid, branch in enumerate(self.stage2)
            ],
                                 dim=3)
        assert output.size(1) == 1
        assert output.size(2) == 1
        assert output.size(3) == len(self.stage2)
        # output : nsample, n_out_hidden, ndoa
        if output.size(3) > 1:
            return output[:, 0, 0]
        else:
            return output[:, 0, 0, 0]

    def forward_feature_output(self, x, no_out_act=False):
        x = self.stage1trunk(x)
        f = torch.cat([branch(x) for branch in self.stage1out], dim=1)
        # feature activation map : ndata, n_out_map * ndoa, nframe, nfbin

        ndata, ndoabranch, nframe, nfbin = f.size()
        nbranch = len(self.stage1out)
        assert self.output_size == ndoabranch // nbranch
        x = f.view((ndata, nbranch, self.output_size, nframe, nfbin))
        # ndata, n_out_map, ndoa, nframe, nfbin
        x = x.permute(0, 4, 3, 1, 2)
        # ndata, nfbin, nframe, n_out_map, ndoa
        o = self._stage2_internal(x, no_out_act)
        return f, o

    def forward_feature(self, x):
        for i in range(self.fl_internal):
            x = self.stage1trunk[i](x)
        return x

    def set_feat_layers(self, feat_layers):
        assert feat_layers + 4 <= len(self.stage1trunk)
        if self <= 2:
            self.fl_internal = 3 * feat_layers
        else:
            self.fl_internal = 4 + feat_layers

    def partial_params(self):
        return itertools.chain(*[
            self.stage1trunk[i].parameters() for i in range(self.fl_internal)
        ])


class SslSnscLoss:
    """ Source localization and speech/non-speech loss:
        L2-Loss for SSL + mu * L2-Loss for SNSC weighed by SSL gt
    """
    def __init__(self, mu, sns_width_factor=1.0):
        self.mu = mu
        self.spf = 1.0 / (sns_width_factor**2.0)

    def __call__(self, pred, gt):
        ndata, nf, ndoa = pred.size()
        ngdata, ngf, ngdoa = gt.size()
        assert nf == 2
        assert ngdata == ndata and ngf == nf and ngdoa == ndoa
        ssl_loss = ((pred[:, 0] - gt[:, 0])**2.0).mean()
        snsc_loss = ((pred[:, 1] - gt[:, 1])**2.0 * gt[:, 0]**self.spf).mean()

        return ssl_loss + self.mu * snsc_loss


class AddConstantSns(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net.forward(x).view(x.size(0), 1, 360)
        sns = torch.ones(out.size())
        if out.is_cuda:
            sns = sns.cuda()
        return torch.cat((out, sns), dim=1)


class ResNetDomainClassifier(SerializableModule):
    def __init__(self, input_size, n_res_blocks=5):
        super().__init__({
            'input_size': input_size,
            'n_res_blocks': n_res_blocks
        })

        ic, x, y = input_size

        # stage one:
        seq = []

        # initial layers (no residual)
        # layer 1
        oc = 4 * ic
        ic = oc
        x = x
        y = (y - 7 + 3) // 3

        # layer 2
        oc = 4 * ic
        ic = oc
        x = x
        y = (y - 5 + 2) // 2

        # residual layers
        for _ in range(n_res_blocks):
            seq.append(ResidualBlock(ic, oc))

        # downsampling layers
        oc = ic // 8
        seq.append(nn.Conv2d(ic, oc, kernel_size=(3, 3), stride=(2, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))
        ic = oc
        x = (x - 3 + 2) // 2
        y = (y - 3 + 3) // 3

        oc = ic // 8
        seq.append(nn.Conv2d(ic, oc, kernel_size=(3, 3), stride=(2, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))
        ic = oc
        x = (x - 3 + 2) // 2
        y = (y - 3 + 3) // 3

        # stage one trunk
        self.conv_module = nn.Sequential(*seq)
        self.out_module = nn.Linear(oc * x * y, 2)

    def forward(self, x):
        x = self.conv_module(x)
        x = x.view(x.size(0), -1)
        return self.out_module(x)


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
