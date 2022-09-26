"""
multitask_ms.py

multi-stage multi-task network

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import torch
import torch.nn as nn

from .base import (SerializableModule, copy_model, CONV_ARG_IN_CHS,
                   CONV_ARG_OUT_CHS, CONV_ARG_KERNEL, CONV_ARG_STRIDE,
                   CONV_ARG_PAD, CONV_ARG_DIAL)
from .multitask_v2 import DoaSingleTaskResnet, DoaResnetTrunk


class DoaConvHourglass(nn.Module):
    """
    Apply downsampling convolution to time-freq. predictions and
    then upsample to the same size by transposed convolution.
    The convlutions are along the time-doa axes.
    """
    def __init__(self, l_ds_conv, l_us_conv, n_doa, roll_padding=True):
        """
        Args:
            l_ds_conv : list of downsampling convolution layer arguments
                        see torch.nn.Conv2d
            l_us_conv : list of upsampling convolution layer arguments
                        see torch.nn.Conv2d
            n_doa  : number of directions-of-arrival
            roll_padding : apply padding so that DOA left and right is
                           connected.  should be True if range of DOA is 360
                           degrees.
        """
        super().__init__()

        # tricky part is roll padding along DOA axis
        seq = []
        n_ch = None
        s_doa = n_doa  # size of DOA at input
        for args in l_ds_conv:
            assert CONV_ARG_DIAL not in args
            assert CONV_ARG_PAD not in args
            assert n_ch is None or args[CONV_ARG_IN_CHS] == n_ch
            n_ch = args[CONV_ARG_OUT_CHS]

            if CONV_ARG_STRIDE in args:
                if isinstance(args[CONV_ARG_STRIDE], tuple):
                    stride_doa = args[CONV_ARG_STRIDE][1]
                else:
                    stride_doa = args[CONV_ARG_STRIDE]
            else:
                stride_doa = 1
            if CONV_ARG_KERNEL in args:
                if isinstance(args[CONV_ARG_KERNEL], tuple):
                    kernel_doa = args[CONV_ARG_KERNEL][1]
                else:
                    kernel_doa = args[CONV_ARG_KERNEL]
            else:
                kernel_doa = 1
            assert stride_doa == 1

            s_doa = s_doa + kernel_doa - 1
            n_ch = args[CONV_ARG_OUT_CHS]

            seq.append(nn.Conv2d(**args))
            seq.append(nn.BatchNorm2d(n_ch))
            seq.append(nn.ReLU(inplace=True))

        for args in l_us_conv:
            if CONV_ARG_STRIDE in args:
                if isinstance(args[CONV_ARG_STRIDE], tuple):
                    stride_doa = args[CONV_ARG_STRIDE][1]
                else:
                    stride_doa = args[CONV_ARG_STRIDE]
            else:
                stride_doa = 1
            if CONV_ARG_KERNEL in args:
                if isinstance(args[CONV_ARG_KERNEL], tuple):
                    kernel_doa = args[CONV_ARG_KERNEL][1]
                else:
                    kernel_doa = args[CONV_ARG_KERNEL]
            else:
                kernel_doa = 1
            assert stride_doa == 1
            assert not roll_padding or kernel_doa == 1

            n_ch = args[CONV_ARG_OUT_CHS]

            seq.append(nn.ConvTranspose2d(**args))
            seq.append(nn.BatchNorm2d(n_ch))
            seq.append(nn.ReLU(inplace=True))

        self.seqential = nn.Sequential(*seq[:-1])  # remove last ReLU

        if roll_padding:
            self.doa_pad_size = s_doa - n_doa
        else:
            self.doa_pad_size = 0

    def forward(self, x):
        """
        Args:
            x : time-doa features, tensor of (data, feature, time, doa)
        """
        # padding
        if self.doa_pad_size > 0:
            x = torch.cat((x, x.narrow(3, 0, self.doa_pad_size)), dim=3)

        # downsampling and upsampling
        return self.seqential(x)


class DoaMultiTaskMultiStage(SerializableModule):
    """
    1. stage : two separate single-task network
    n. stage : merge predictions of all tasks and predict differential
               (with residual connection)

    specifically, output at stage s of task t is:
    f_t^(s)(x) = h_t (g_t^(s))
    and g is the intermediate time-frequency local prediction of f, and it is:
    | g_t^(1)(x) = g_t(x)
    | g_t^(s)(x) = g_t^(s-1)(x) + m_t^(s)(g_1^(s-1)(x), g_2^(s-1)(x), ..., g_T^(s-1)(x)), s > 1
    and m is the modification function
    """
    def __init__(self, n_stage, l_st_args, l_m_args, shared_args=None):
        """
        Args:
            n_stage : number of stages
            l_st_args : list (per task) of DoaSingleTaskResnet arguments
            l_m_args : list (per task) of merging convolutions
                       (DoaConvHourglass) arguments
            shared_args : if not None, add a shared feature extraction
                          (DoaResnetTrunk)
                          that is concatenated to the refinement function input
        """
        super().__init__({
            'n_stage': n_stage,
            'l_st_args': l_st_args,
            'l_m_args': l_m_args,
            'shared_args': shared_args,
        })

        # check number of tasks
        assert len(l_st_args) == len(l_m_args)

        self.n_stage = n_stage
        self.n_task = len(l_st_args)
        self.l_fg = nn.ModuleList(
            [DoaSingleTaskResnet(**args) for args in l_st_args])
        self.l_m = nn.ModuleList([
            DoaConvHourglass(**args) for _ in range(n_stage - 1)
            for args in l_m_args
        ])
        self.relu = nn.ReLU(inplace=True)

        if shared_args is not None:
            self.shared = DoaResnetTrunk(**shared_args)
        else:
            self.shared = None

    def forward(self, x, s=None, out_g=False, out_all=False):
        """
        Args:
            x : input tensor, indices (data, channel, time, freq)
            s : number of stages (to replace network stages)
            out_g : output g (time-frequency local prediction) instead of f
            out_all : output predictions of all orders as a list

        Returns:
            if not out_all:
                list of predictions of each task
                [f_1^(s), ..., f_T^(s)]
            else:
                list of list of predictions of each task at all stages
                [
                    [f_1^(1), ..., f_T^(1)],
                    ...
                    [f_1^(s), ..., f_T^(s)]
                ]
            and if out_g:
                replace f with g
        """
        if s is None:
            s = self.n_stage
        assert s > 0 and s <= self.n_stage, f'Invalid stage {s}/{self.n_stage}'

        l_g_s = []  # list (per stage) of g
        l_f_s = []  # list (per stage) of f

        # get shared feature (if necessary)
        if self.shared is not None:
            l_shared = [self.shared(x)]
        else:
            l_shared = []

        # 1. stage
        # stage1=True to get t-f local pred (i.e. g)
        l_g = [fg.forward_stage1(x) for fg in self.l_fg]
        l_g_s.append(l_g)

        # iter s - 1 times
        for i in range(s - 1):
            # still previous stage
            # check if f need to be computed for this stage
            if out_all and not out_g:
                # compute f
                l_f_s.append(
                    [fg.forward_stage2(g) for fg, g in zip(self.l_fg, l_g)])

            # compute next g
            # input for m, concat all g
            # g : (data, freq, time, doa)
            # concatenate along freq axis
            gcc = torch.cat(l_g + l_shared, dim=1)
            # compute for all tasks
            l_g = [
                self.relu(l_g[t] + self.l_m[self.n_task * i + t](gcc))
                for t in range(self.n_task)
            ]
            l_g_s.append(l_g)

        if out_g:
            if out_all:
                return l_g_s
            else:
                return l_g
        else:
            # compute last f
            l_f_s.append(
                [fg.forward_stage2(g) for fg, g in zip(self.l_fg, l_g)])
            if out_all:
                return l_f_s
            else:
                return l_f_s[-1]

    def init_pretrained_stage1(self, src):
        """
        Initialize using pre-trained stage 1 (i.e. g_t and h_t for any task t)
        parameters.

        Args:
            src : source (pre-trained) model
        """
        assert type(src) == type(self)
        assert src.n_stage == 1
        assert src.n_task == self.n_task
        for t in range(self.n_task):
            copy_model(self.l_fg[t], src.l_fg[t])
