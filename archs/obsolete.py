"""
obsolete.py

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import torch
import torch.nn as nn

from .base import SerializableModule, ResidualBlock, _act_funcs, ACT_NONE, ACT_SIGMOID, ACT_RELU


class ResNet(SerializableModule):
    def __init__(self,
                 input_size,
                 output_act=ACT_NONE,
                 n_out_map=1,
                 n_out_hidden=0,
                 output_size=360):
        super().__init__({
            'input_size': input_size,
            'output_act': output_act,
            'n_out_map': n_out_map,
            'n_out_hidden': n_out_hidden,
            'output_size': output_size
        })

        self.output_size = output_size

        ic, x, y = input_size

        # conv layers
        seq = []

        # initial layers (no residual)
        # layer 1
        oc = 4 * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 7), stride=(1, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 7 + 3) // 3

        # layer 2
        oc = 4 * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 5), stride=(1, 2)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 5 + 2) // 2

        # residual layers
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))

        # reduce map size
        oc = ic / 4
        seq.append(nn.Conv2d(ic, oc, kernel_size=(x, 9), stride=(1, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = 1
        y = (y - 9 + 3) // 3

        self.cseq = nn.Sequential(*seq)

        # output layers
        if n_out_hidden == 0:
            outseq = [nn.Linear(ic * x * y, output_size * n_out_map)]
            if output_act != ACT_NONE:
                outseq.append(_act_funcs[output_act]())
            self.out = nn.Sequential(*outseq)
        else:
            hidden_struct = [1000] * n_out_hidden + [output_size]
            self.out = MultiTaskMLP(ic * x * y, [hidden_struct] * n_out_map,
                                    hidden_act=ACT_RELU,
                                    output_act=output_act,
                                    batch_norm=True)

    def forward(self, x):
        x = self.cseq(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output.view(x.size(0), -1, self.output_size)


class ResNetv2(SerializableModule):
    """Expected map sizes:
        15x168
            -> conv(1,5) stride(1,3)
        15x55
            -> conv(1,3) stride(1,2)
        15x27
            -> residual modules
        15x27
            -> conv(3,3) stride(2,2)
         7x13
            -> conv(3,3) stride(2,2)
         3x6
            -> fully connected
        1000
            -> fully connected
         360
    """
    def __init__(self,
                 input_size,
                 init_ch_factor=4,
                 output_act=ACT_NONE,
                 n_out_map=1,
                 n_out_hidden=0,
                 output_size=360):
        super().__init__({
            'input_size': input_size,
            'init_ch_factor': init_ch_factor,
            'output_act': output_act,
            'n_out_map': n_out_map,
            'n_out_hidden': n_out_hidden,
            'output_size': output_size
        })

        self.output_size = output_size

        ic, x, y = input_size

        # conv layers
        seq = []

        # initial layers (no residual)
        # layer 1
        oc = init_ch_factor * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 5), stride=(1, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 5 + 3) // 3

        # layer 2
        oc = 4 * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 3), stride=(1, 2)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 3 + 2) // 2

        # residual layers
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))

        # reduce map size 1
        oc = ic / 4
        seq.append(nn.Conv2d(ic, oc, kernel_size=(3, 3), stride=(2, 2)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = (x - 3 + 2) // 2
        y = (y - 3 + 2) // 2

        # reduce map size 2
        oc = ic / 4
        seq.append(nn.Conv2d(ic, oc, kernel_size=(3, 3), stride=(2, 2)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = (x - 3 + 2) // 2
        y = (y - 3 + 2) // 2

        self.cseq = nn.Sequential(*seq)

        # output layers
        if n_out_hidden == 0:
            outseq = [nn.Linear(ic * x * y, output_size * n_out_map)]
            if output_act != ACT_NONE:
                outseq.append(_act_funcs[output_act]())
            self.out = nn.Sequential(*outseq)
        else:
            hidden_struct = [1000] * n_out_hidden + [output_size]
            self.out = MultiTaskMLP(ic * x * y, [hidden_struct] * n_out_map,
                                    hidden_act=ACT_RELU,
                                    output_act=output_act,
                                    batch_norm=True)

    def forward(self, x):
        x = self.cseq(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output.view(x.size(0), -1, self.output_size)


class ResNetCtx32(SerializableModule):
    def __init__(self,
                 input_size,
                 output_act=ACT_NONE,
                 n_out_map=1,
                 n_out_hidden=0):
        super().__init__({
            'input_size': input_size,
            'output_act': output_act,
            'n_out_map': n_out_map,
            'n_out_hidden': n_out_hidden
        })

        ic, x, y = input_size

        # conv layers
        seq = []

        # initial layers (no residual)
        # layer 1
        oc = 4 * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 7), stride=(1, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 7 + 3) // 3

        # layer 2
        oc = 4 * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(2, 5), stride=(2, 2)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x // 2
        y = (y - 5 + 2) // 2

        # residual layers
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))

        # reduce map size
        oc = ic / 8
        seq.append(nn.Conv2d(ic, oc, kernel_size=(4, 9), stride=(4, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x // 4
        y = (y - 9 + 3) // 3

        self.cseq = nn.Sequential(*seq)

        # output layers
        if n_out_hidden == 0:
            outseq = [nn.Linear(ic * x * y, 360 * n_out_map)]
            if output_act != ACT_NONE:
                outseq.append(_act_funcs[output_act]())
            self.out = nn.Sequential(*outseq)
        else:
            hidden_struct = [1000] * n_out_hidden + [360]
            self.out = MultiTaskMLP(ic * x * y, [hidden_struct] * n_out_map,
                                    hidden_act=ACT_RELU,
                                    output_act=output_act,
                                    batch_norm=True)

    def forward(self, x):
        x = self.cseq(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output.view(x.size(0), -1, 360)


class MLP_Softmax(SerializableModule):
    def __init__(self, layer_size, hidden_act=ACT_SIGMOID, batch_norm=False):
        super().__init__({
            'layer_size': layer_size,
            'hidden_act': hidden_act,
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
            else:
                seq.append(nn.Softmax())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x.view(x.size(0), -1))


class ResNetClassification(SerializableModule):
    def __init__(self, input_size, output_size, channel_x=1):
        super().__init__({
            'input_size': input_size,
            'output_size': output_size,
            'channel_x': channel_x
        })

        x, y = input_size
        ic = 1

        # conv layers
        seq = []

        # initial layers (no residual)
        # layer 1
        oc = 8 * ic * channel_x
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 7), stride=(1, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 7 + 3) // 3

        # layer 2
        oc = 8 * ic
        seq.append(nn.Conv2d(ic, oc, kernel_size=(1, 5), stride=(1, 2)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = x
        y = (y - 5 + 2) // 2

        # residual layers
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))
        seq.append(ResidualBlock(ic, oc))

        # reduce map size
        oc = ic / 4
        seq.append(nn.Conv2d(ic, oc, kernel_size=(x, 9), stride=(1, 3)))
        seq.append(nn.BatchNorm2d(oc))
        seq.append(nn.ReLU(inplace=True))

        ic = oc
        x = 1
        y = (y - 9 + 3) // 3

        self.cseq = nn.Sequential(*seq)

        # output layers
        outseq = [nn.Linear(ic * x * y, output_size)]
        outseq.append(nn.Softmax())
        self.out = nn.Sequential(*outseq)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cseq(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4
