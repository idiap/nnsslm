"""
multitask_v2.py

Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import torch
import torch.nn as nn

from .base import (SerializableModule, copy_model, ResidualBlock, _act_funcs,
                   ACT_NONE, POOL_MAX, POOL_AVG, POOL_NONE, CONV_ARG_IN_CHS,
                   CONV_ARG_OUT_CHS, CONV_ARG_KERNEL, CONV_ARG_STRIDE,
                   CONV_ARG_PAD, CONV_ARG_DIAL)


class AbstractDoaResent(object):
    """
    Abstract class of network for DOA-wise prediction with ResNet trunk.
    """
    def __init__(self, n_freq, n_doa, roll_padding, train_no_pool):
        """
        Args:
            n_freq : number of frequency bins
            n_doa  : number of directions-of-arrival
            roll_padding : apply padding so that DOA left and right is
                           connected.  should be True if range of DOA is 360
                           degrees.
            train_no_pool : do not apply pooling along time during training
        """
        self.n_freq = n_freq
        self.n_doa = n_doa
        self.roll_padding = roll_padding
        self.train_no_pool = train_no_pool

    def construct_trunk(self, ds_convs, n_rblk):
        """
        Construct the trunk, i.e. down-sampling and residual blocks

        Args:
            ds_convs : downsampling layers at the beginning of the network,
                       list of arguments (in dict) for creating nn.Conv2d.
            n_rblk : number of residual blocks in the shared layers

        Returns:
            trunk module
        """
        seq = []
        n_ch = None
        n_freq = self.n_freq  # working copy of # of freq.
        for l in ds_convs:
            assert CONV_ARG_DIAL not in l
            assert CONV_ARG_PAD not in l
            assert n_ch is None or l[CONV_ARG_IN_CHS] == n_ch
            n_ch = l[CONV_ARG_OUT_CHS]

            seq.append(nn.Conv2d(**l))
            seq.append(nn.BatchNorm2d(n_ch))
            seq.append(nn.ReLU(inplace=True))

            # compute map size along frequency
            if CONV_ARG_STRIDE in l:
                if isinstance(l[CONV_ARG_STRIDE], tuple):
                    stride_freq = l[CONV_ARG_STRIDE][1]
                else:
                    stride_freq = l[CONV_ARG_STRIDE]
            else:
                stride_freq = 1
            if CONV_ARG_KERNEL in l:
                if isinstance(l[CONV_ARG_KERNEL], tuple):
                    kernel_freq = l[CONV_ARG_KERNEL][1]
                else:
                    kernel_freq = l[CONV_ARG_KERNEL]
            else:
                kernel_freq = 1
            n_freq = (n_freq - kernel_freq + stride_freq) // stride_freq

        # residual layers
        for _ in range(n_rblk):
            seq.append(ResidualBlock(n_ch, n_ch))

        # n_ch and n_freq at trunk output
        self.n_ch_trunk = n_ch
        self.n_freq_trunk = n_freq
        return nn.Sequential(*seq)

    def construct_doa_mapper(self, df_convs):
        """
        Construct mapping to DOA (and time, i.e. remove freq and add DOA).

        Args:
            df_convs : (doa-feature) convolution layers after the feature is
                        mapped to DOA.
                        if None, assume no padding.

        Returns:
            to_doa : module of DOA mapping
            pad_size : size for roll padding
        """
        # compute doa padding size
        s_doa = self.n_doa
        # s_doa : expected size on DOA after to_doa
        #         s_doa depends on the size change of the features along DOA
        #         axis during the convolutions.
        if df_convs is not None:
            for l in df_convs[::-1]:
                assert CONV_ARG_DIAL not in l
                assert CONV_ARG_PAD not in l

                if CONV_ARG_STRIDE in l:
                    if isinstance(l[CONV_ARG_STRIDE], tuple):
                        stride_doa = l[CONV_ARG_STRIDE][1]
                    else:
                        stride_doa = l[CONV_ARG_STRIDE]
                else:
                    stride_doa = 1
                if CONV_ARG_KERNEL in l:
                    if isinstance(l[CONV_ARG_KERNEL], tuple):
                        kernel_doa = l[CONV_ARG_KERNEL][1]
                    else:
                        kernel_doa = l[CONV_ARG_KERNEL]
                else:
                    kernel_doa = 1
                assert stride_doa == 1
                s_doa = s_doa + kernel_doa - 1

        # construct module and return
        if self.roll_padding:
            return (
                nn.Sequential(
                    nn.Conv2d(self.n_ch_trunk, self.n_doa, kernel_size=1),
                    nn.BatchNorm2d(self.n_doa),
                    nn.ReLU(inplace=True),
                ),
                s_doa - self.n_doa,
            )
        else:
            return (
                nn.Sequential(
                    nn.Conv2d(self.n_ch_trunk, s_doa, kernel_size=1),
                    nn.BatchNorm2d(s_doa),
                    nn.ReLU(inplace=True),
                ),
                0,
            )

    def get_doawise_feat_size(self):
        """
        Get DOA-wise feature size

        Returns:
            DOA-wise feature size
        """
        return self.n_freq_trunk

    def construct_df_convs(self, df_convs, out_act):
        """
        Construct module of convlutions along time and DOA.

        Args:
            df_convs : (doa-feature) convolution layers after the feature is
                        mapped to DOA.
            out_act : output activation function (see .basic.ACT_*)

        Returns:
            to_doa : module of DOA mapping
            pad_size : size for roll padding
        """
        seq = []
        n_ch = self.get_doawise_feat_size()
        for i, l in enumerate(df_convs):
            assert l[CONV_ARG_IN_CHS] == n_ch
            n_ch = l[CONV_ARG_OUT_CHS]

            seq.append(nn.Conv2d(**l))
            if i < len(df_convs) - 1:
                seq.append(nn.BatchNorm2d(n_ch))
                seq.append(nn.ReLU(inplace=True))
            else:
                seq.append(_act_funcs[out_act]())
        return nn.Sequential(*seq)

    def apply_pooling(self, y, m):
        """
        Apply pooling along time axis, only if not (training and train_no_pool)

        Args:
            y : tensor to be pooled, indices (data, feature, time, doa)
            m : pooling method

        Returns:
            pooled tensor, indices
                - data, feature, doa         (if pooling)
                - data, feature, time, doa   (if no pooling)
        """
        # check whether apply pooling
        if self.training and self.train_no_pool:
            m = POOL_NONE
        # now apply
        if m == POOL_AVG:
            # average pooling
            y = torch.mean(y, 2)
        elif m == POOL_MAX:
            # max pooling
            y = torch.max(y, 2)[0]
        elif m == POOL_NONE:
            pass
        else:
            assert False
        return y

    def reformat_output(self, y):
        """
        reformat y to:
            (data, doa)                   : if prediction is 1-dim and pool
            (data, doa, time)             : if prediction is 1-dim and no pool
            (data, doa, prediction)       : if prediction is n-dim and pool
            (data, doa, time, prediction) : if prediction is n-dim and no pool
        Args:
            y : tensor

        Return:
            reformatted y
        """
        #   swap : feature to last dim
        if y.dim() == 3:
            y = y.permute(0, 2, 1)
            # result : data, doa, feature
        else:
            assert y.dim() == 4
            y = y.permute(0, 2, 3, 1)
            # result : data, time, doa, feature

        #   squeeze if feature dimension is one
        if y.size(-1) == 1:
            y = y.squeeze(-1)
            # result : data, doa         (if pooling)
            # result : data, time, doa   (if no pooling)

        return y


class DoaMultiTaskResnet(SerializableModule, AbstractDoaResent):
    def __init__(self,
                 n_freq,
                 n_doa,
                 ds_convs,
                 n_rblk,
                 task_layers,
                 roll_padding=True,
                 train_no_pool=False):
        """
        Args:
            n_freq : number of frequency bins
            n_doa  : number of directions-of-arrival
            ds_convs : downsampling layers at the beginning of the network,
                       list of arguments (in dict) for creating nn.Conv2d.
            n_rblk : number of residual blocks in the shared layers
            task_layers : list of task-specific layers, each branch is
                          represented by convolution layers, the output
                          activation function, and the pooling (along time)
                          method
            roll_padding : apply padding so that DOA left and right is
                           connected.  should be True if range of DOA is 360
                           degrees.
            train_no_pool : do not apply pooling along time during training
        """
        super().__init__({
            'n_freq': n_freq,
            'n_doa': n_doa,
            'ds_convs': ds_convs,
            'n_rblk': n_rblk,
            'task_layers': task_layers,
            'roll_padding': roll_padding,
            'train_no_pool': train_no_pool,
        })
        AbstractDoaResent.__init__(self, n_freq, n_doa, roll_padding,
                                   train_no_pool)

        # trunk
        self.shared = self.construct_trunk(ds_convs, n_rblk)

        # task-specific layers
        to_doas = []
        branches = []
        self.pad_size = []
        self.pool_mthd = []
        for layers, act, pool_mthd in task_layers:
            # pooling method
            self.pool_mthd.append(pool_mthd)

            to_doa, pad_size = self.construct_doa_mapper(layers)
            to_doas.append(to_doa)
            self.pad_size.append(pad_size)

            # convolution layers
            branches.append(self.construct_df_convs(layers, act))
        self.to_doas = nn.ModuleList(to_doas)
        self.branches = nn.ModuleList(branches)

    def forward(self, x, stage1=False, stage12=False):
        """
        Per task output:
            Stage-1 : after DOA-mapper, tensor of (data, freq, time, doa)
            Stage-2 : see AbstractDoaResent.reformat_output

        Args:
            x : input tensor of (data, ch, time, freq)
            stage1 : return stage-1 output instead of final (stage-2)
            stage12 : return both stage-1 and stage-2 (as a tuple)

        Returns:
            list of results (per task), result see arguments
        """
        assert not (stage1 and stage12), 'conflicting output requests'

        # forward stage1
        s1res = self._forward_stage1(x)

        # return if only stage 1 result is requested
        if stage1:
            return s1res

        # forward stage2
        s2res = self._forward_stage2(s1res)

        if stage12:
            return s1res, s2res
        else:
            return s2res

    def _forward_stage1(self, x):
        # input : data, ch, time, freq
        x = self.shared(x)
        # now   : data, feature, time, freq

        s1res = []  # stage-1 result

        # for each branch
        for t in self.to_doas:
            # make doa as feature
            y = t(x)
            # now  : data, doa, time, freq
            # swap : freq <-> doa
            y = y.permute(0, 3, 2, 1)
            # now  : data, freq, time, doa
            # this is stage-1 result
            s1res.append(y)
        return s1res

    def _forward_stage2(self, s1res):
        s2res = []  # stage-2 result
        for y, b, p, m in zip(s1res, self.branches, self.pad_size,
                              self.pool_mthd):
            # padding
            if p > 0:
                y = torch.cat((y, y.narrow(3, 0, p)), dim=3)
            # now  : data, freq, time, doa
            # more convolutions
            y = b(y)
            # now  : data, feature, time, doa

            # pooling : along time
            y = self.apply_pooling(y, m)
            # now  : data, feature, doa         (if pooling)
            # now  : data, feature, time, doa   (if no pooling)

            # change output format
            y = self.reformat_output(y)

            s2res.append(y)
        return s2res

    def forward_stage1(self, x):
        return self._forward_stage1(x)

    def forward_stage2(self, x):
        return self._forward_stage2(x)

    def strip_last_layer(self, task_id):
        """
        Make a copy of the network with the last layer of a branch removed.
        It is useful to get speaker embedding from speaker identification
        models.

        Args:
            task_id : the branch of which the last layer is to be stripped

        Returns:
            copy of the model with specified layer removed
        """
        # construct model
        n_task_layers = self.arch_args['task_layers'].copy()
        layers, act, pool_mthd = n_task_layers[task_id]
        n_layers = layers.copy()
        n_layers.pop()
        n_task_layers[task_id] = (n_layers, act, pool_mthd)
        n_args = self.arch_args.copy()
        n_args['task_layers'] = n_task_layers
        n_model = DoaMultiTaskResnet(**n_args)

        # copy parameters to the new model
        # assume names of the parameters stay the same
        # and the parameters of the new model is a subset of the old one
        copy_model(n_model, self)

        return n_model


class DoaSingleTaskResnet(DoaMultiTaskResnet):
    """
    single-task case of DoaMultiTaskResnet.

    Forward returns single prediction instead of a list (as DoaMultiTaskResnet)
    """
    def __init__(self,
                 n_freq,
                 n_doa,
                 ds_convs,
                 n_rblk,
                 df_convs,
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
            out_act : output activation function (see .basic.ACT_*)
            out_pool : pooling (along time) method at output
                       (see .basic.POOL_*)
            roll_padding : apply padding so that DOA left and right is
                           connected. Should be True if range of DOA is 360
                           degrees.
            train_no_pool : do not apply pooling along time during training
        """
        super().__init__(
            n_freq,
            n_doa,
            ds_convs,
            n_rblk,
            [(df_convs, out_act, out_pool)],
            roll_padding=roll_padding,
            train_no_pool=train_no_pool,
        )
        # overwrite parents' args
        self.arch_args = {
            'n_freq': n_freq,
            'n_doa': n_doa,
            'ds_convs': ds_convs,
            'n_rblk': n_rblk,
            'df_convs': df_convs,
            'out_act': out_act,
            'out_pool': out_pool,
            'roll_padding': roll_padding,
            'train_no_pool': train_no_pool,
        }

    def forward(self, x, **kargs):
        """
        Returns:
            pred : predicted tensor
        """
        res = super().forward(x, **kargs)
        if type(res) == list:
            assert len(res) == 1
            return res[0]
        else:
            assert type(res) == tuple
            # stage 1 and 2 results
            s1res, s2res = res
            assert len(s1res) == 1
            assert len(s2res) == 1
            return s1res[0], s2res[0]

    def forward_stage1(self, x):
        return self.forward(x, stage1=True)

    def forward_stage2(self, s1res):
        return super().forward_stage2([s1res])[0]


class DoaResnetTrunk(AbstractDoaResent, nn.Module):
    """
    Trunk of DoaSingleTaskResnet/DoaMultiTaskResnet.
    """
    def __init__(self,
                 n_freq,
                 n_doa,
                 ds_convs,
                 n_rblk,
                 roll_padding=True,
                 feat_size=1):
        """
        Args:
            n_freq : number of frequency bins
            n_doa  : number of directions-of-arrival
            ds_convs : downsampling layers at the beginning of the network,
                       list of arguments (in dict) for creating nn.Conv2d.
            n_rblk : number of residual blocks in the shared layers
            roll_padding : apply padding so that DOA left and right is
                           connected. should be True if range of DOA is 360
                           degrees.
            feat_size : feature size per time, doa, freq
        """
        if not roll_padding:
            raise NotImplementedError(
                'DoaMultiTaskResnet does not support roll_padding=False')

        super().__init__(n_freq, n_doa * feat_size, roll_padding, None)
        nn.Module.__init__(self)

        self.feat_size = feat_size

        # trunk
        self.shared = self.construct_trunk(ds_convs, n_rblk)

        # task-specific layers
        self.to_doa, _ = self.construct_doa_mapper(None)

    def forward(self, x):
        """
        Args:
            x : input tensor of (data, ch, time, freq)

        Returns:
            y : output tensor of (data, feature, time, doa)
        """
        # input : data, ch, time, freq
        y = self.shared(x)
        # now   : data, feature, time, freq

        # map to DOA
        y = self.to_doa(y)
        # now  : data, feature * doa, time, freq

        # separate feature and doa
        n_data, _, n_time, n_freq = y.size()
        y = y.reshape((n_data, self.feat_size, self.n_doa // self.feat_size,
                       n_time, n_freq))
        # now  : data, feature, doa, time, freq

        # swap : freq <-> doa
        y = y.permute(0, 1, 4, 3, 2)
        # now  : data, feature, freq, time, doa

        # merge feature and freq
        y = y.reshape((n_data, self.feat_size * n_freq, n_time,
                       self.n_doa // self.feat_size))
        return y
