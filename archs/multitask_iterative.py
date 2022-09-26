"""
multitask_iterative.py

Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

from itertools import chain

import torch
import torch.nn as nn

from .base import (SerializableModule, ACT_NONE, POOL_NONE, POOL_AVG, POOL_MAX,
                   CONV_ARG_IN_CHS, CONV_ARG_OUT_CHS)

from .multitask_v2 import AbstractDoaResent


class DoaSingleTaskAuxInput(SerializableModule, AbstractDoaResent):
    """
    DOA-wise prediction network for single task with auxiliary input.
    The auxiliary input, which is ground truth or prediction of other tasks,
    are concatenated to the features that are mapped to DOA.
    """
    def __init__(self,
                 n_freq,
                 n_doa,
                 ds_convs,
                 n_rblk,
                 df_convs,
                 au_tconvs=None,
                 out_act=ACT_NONE,
                 out_pool=POOL_NONE,
                 roll_padding=True,
                 train_no_pool=False):
        """
        Args:
            n_freq : number of frequency bins
            n_doa  : number of directions-of-arrival
            ds_convs : downsampling layers at the beginning of the network,
                       list of arguments (in dict) for creating nn.Conv2d.
            n_rblk : number of residual blocks in the shared layers
            df_convs : (doa-feature) convolution layers after the feature is
                        mapped to DOA.
            au_tconvs : convolutions layers for the auxiliary input, in general
                       it should map the input to the same size (time axis)
                       as the DOA-wise features (after DOA mapper).
            out_act : output activation function (see .basic.ACT_*)
            out_pool : pooling (along time) method at output (see .basic.POOL_*)
            roll_padding : apply padding so that DOA left and right is connected.
                           should be True if range of DOA is 360 degrees.
            train_no_pool : do not apply pooling along time during training
        """
        super().__init__({
            'n_freq': n_freq,
            'n_doa': n_doa,
            'ds_convs': ds_convs,
            'n_rblk': n_rblk,
            'df_convs': df_convs,
            'au_tconvs': au_tconvs,
            'out_act': out_act,
            'out_pool': out_pool,
            'roll_padding': roll_padding,
            'train_no_pool': train_no_pool,
        })
        AbstractDoaResent.__init__(self, n_freq, n_doa, roll_padding,
                                   train_no_pool)
        self.out_pool = out_pool

        # trunk
        self.trunk = self.construct_trunk(ds_convs, n_rblk)

        # auxiliary branch
        self.aux_mapper, self.aux_feat_size = self.construct_aux_mapper(
            au_tconvs)

        # doa mapper
        self.to_doa, self.pad_size = self.construct_doa_mapper(df_convs)

        # df conv module
        self.df_module = self.construct_df_convs(df_convs, out_act)

    def forward(self, x, z=None):
        """
        Args:
            x : input tensor, indices (data, channel, time, freq)
            z : auxiliary input tensor, indices (data, time, doa, feature)
                if z is None, this is basically degraded to single task network,
                i.e. no auxiliary input is used.
        """
        # input : data, ch, time, freq
        x = self.trunk(x)
        # now   : data, feature, time, freq

        # make doa as feature
        y = self.to_doa(x)
        # now  : data, doa, time, freq
        # swap : freq <-> doa
        y = y.permute(0, 3, 2, 1)
        # now  : data, freq, time, doa

        # concat z to y
        if z is not None:
            # z : data, time, doa, feature
            # swap axes
            z = z.permute(0, 3, 1, 2)
            # z : data, feature, time, doa
            # mapping
            z = self.aux_mapper(z)
            # concatenate
            y = torch.cat((y, z), dim=1)

        # padding
        if self.pad_size > 0:
            y = torch.cat((y, y.narrow(3, 0, self.pad_size)), dim=3)
        # now  : data, freq, time, doa
        # more convolutions
        y = self.df_module(y)
        # now  : data, feature, time, doa

        # pooling : along time
        y = self.apply_pooling(y, self.out_pool)
        # now  : data, feature, doa         (if pooling)
        # now  : data, feature, time, doa   (if no pooling)

        # change output format
        y = self.reformat_output(y)

        return y

    def get_doawise_feat_size(self):
        """
        Get DOA-wise feature size

        Returns:
            DOA-wise feature size
        """
        return self.n_freq_trunk + self.aux_feat_size

    def construct_aux_mapper(self, au_tconvs):
        """
        Construct the branch that maps the auxiliary input to the same size as
        the DOA-wise feature so that they can be merged.

        Args:
            au_tconvs : convolutions layers for the auxiliary input, in general
                       it should map the input to the same size (time axis)
                       as the DOA-wise features (after DOA mapper).

        Returns:
            aux_mapper : pytorch module
            out_ch     : output feature size
        """
        if au_tconvs is None:
            return None, 0

        seq = []
        n_ch = None
        for i, l in enumerate(au_tconvs):
            assert n_ch is None or l[CONV_ARG_IN_CHS] == n_ch
            n_ch = l[CONV_ARG_OUT_CHS]

            seq.append(nn.ConvTranspose2d(**l))
            seq.append(nn.BatchNorm2d(n_ch))
            seq.append(nn.ReLU(inplace=True))

        return nn.Sequential(*seq), n_ch


def _expand_4d(y):
    if y.dim() == 3:
        y = y.unsqueeze(-1)
    return y


class DoaMultiTaskIterative(SerializableModule):
    """
    Alternatively and iteratively calling multiple DoaSingleTaskAuxInput.

    Specifically, there are T single task networks with auxiliary input:
        f_1, ..., f_T
    where T is the number of tasks.
    And, T-1 single task network (no auxiliary input) for initial prediction:
        g_1, ..., g_{T-1}

    For an input x, the network computes as follows:

    order 1:
        y_1^(1) = g_1(x)
        ...
        y_{T-1}^(1) = g_{T-1}(x)
        y_T^(1) = f_T(x, y_{1:T-1})

    order o (o > 1):
        y_1^(o) = f_1(x, y_{2:T})
        ...
        y_T^(o) = f_T(x, y_{1:T-1})
    """
    def __init__(self,
                 l_f_args,
                 l_g_args,
                 l_out_pool,
                 default_order=None,
                 train_no_pool=False):
        """
        Args:
            l_f_args : list of single-task network arguments
                       (see DoaSingleTaskAuxInput)
            l_g_args : list of initialization network arguments
                       (see DoaSingleTaskAuxInput)
            l_out_pool : list of pooling (along time) method at output for each
                         task (see .basic.POOL_*)
            default_order : default number of iterations
            train_no_pool : do not apply pooling along time during training
        """
        super().__init__({
            'l_f_args': l_f_args,
            'l_g_args': l_g_args,
            'l_out_pool': l_out_pool,
            'default_order': default_order,
            'train_no_pool': train_no_pool,
        })
        self.l_out_pool = l_out_pool
        self.train_no_pool = train_no_pool

        assert len(l_g_args) == len(l_f_args) - 1
        for args in chain(l_f_args, l_g_args):
            assert 'out_pool' not in args or args['out_pool'] == POOL_NONE

        self.fs = nn.ModuleList(
            [DoaSingleTaskAuxInput(**args) for args in l_f_args])
        self.gs = nn.ModuleList(
            [DoaSingleTaskAuxInput(**args) for args in l_g_args])
        self.default_order = default_order

    def forward(self, x, order=None, out_all=False):
        """
        Args:
            x : input tensor, indices (data, channel, time, freq)
            order : number of iterations, if None use default_order
            out_all : output predictions of all orders as a list

        Returns:
            if not out_all:
                [y_1^(o), ..., y_T^(o)]
                list of predictions of each task at the given order
            else:
                [
                    [y_1^(1), ..., y_T^(1)],
                    ...
                    [y_1^(o), ..., y_T^(o)]
                ]
                list of list of predictions of each task at all orders
        """

        if order is None:
            order = self.default_order
        assert order is not None

        res = []

        for o in range(order):
            if o == 0:
                ys = [_expand_4d(g(x)) for g in self.gs]
                z = torch.cat(ys)
                ys.append(_expand_4d(self.fs[-1](x, z)))
            else:
                yp = ys  # previous order
                ys = []
                for f in self.fs:
                    yp.pop(0)
                    z = torch.cat(tuple(chain(ys, yp)))
                    assert z.dim() == 4
                    ys.append(_expand_4d(f(x, z)))
            res.append([
                self.apply_pooling_and_squeeze(y, p)
                for y, p in zip(ys, self.l_out_pool)
            ])

        if out_all:
            return res
        else:
            return res[-1]

    def apply_pooling_and_squeeze(self, y, m):
        """
        Apply pooling along time axis, only if not (training and train_no_pool)

        Args:
            y : tensor to be pooled, indices (data, time, doa, feature)
            m : poolong method

        Returns:
            pooled tensor, indices
                - data, doa, feature         (if pooling)
                - data, time, doa, feature   (if no pooling)
                (feature dim. is removed if size is 1)
        """
        # check whether apply pooling
        if self.training and self.train_no_pool:
            m = POOL_NONE
        # now apply
        if m == POOL_AVG:
            # average pooling
            y = torch.mean(y, 1)
        elif m == POOL_MAX:
            # max pooling
            y = torch.max(y, 1)[0]
        elif m == POOL_NONE:
            pass
        else:
            assert False

        #   squeeze if feature dimension is one
        if y.size(-1) == 1:
            y = y.squeeze(-1)
            # result : data, doa         (if pooling)
            # result : data, time, doa   (if no pooling)

        return y
