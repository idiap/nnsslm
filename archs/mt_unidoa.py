"""
mt_unidoa.py

uniform DOA feature mapping

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

from .base import (SerializableModule, ResidualBlock, get_conv1d_sequence,
                   get_conv2d_sequence, _GLOBAL_POOL)

import torch.nn as nn


class _DoaMapper(nn.Module):
    """
    Map T-F features to T-D
    """
    def __init__(self, n_freq, n_doa, n_in_feat, tfd_factor):
        super().__init__()
        self.n_doa = n_doa
        self.tfd_factor = tfd_factor

        self.cseq = nn.Sequential(
            nn.Conv2d(n_in_feat, tfd_factor * n_doa, kernel_size=1),
            nn.BatchNorm2d(tfd_factor * n_doa),
            nn.ReLU(),
        )

    def forward(self, x):
        # x : samples, in_feature, time, freq
        y = self.cseq(x)
        # y : samples, tfd_factor * doa, time, freq
        y = y.reshape(y.size(0), self.tfd_factor, self.n_doa, y.size(2),
                      y.size(3))
        # y : samples, tfd_factor, doa, time, freq
        y = y.permute(0, 1, 4, 3, 2)
        # y : samples, tfd_factor, freq, time, doa
        y = y.reshape(y.size(0), y.size(1) * y.size(2), y.size(3), y.size(4))
        # y : samples, tfd_factor * freq, time, doa
        return y


class _FrameWiseBranch(nn.Module):
    """
    Frame-wise prediction branch
    """
    def __init__(self, conv_args, output_act, circular_doa=True):
        super().__init__()
        self.convs, _, _ = get_conv2d_sequence(conv_args,
                                               output_act=output_act,
                                               circular_axis=1)

    def forward(self, x):
        return self.convs(x)


class _SequenceWiseBranch(nn.Module):
    """
    Sequence-wise prediction branch
    """
    def __init__(self,
                 conv_args,
                 pool,
                 mlp_args,
                 output_act,
                 circular_doa=True):
        super().__init__()
        self.convs, _, _ = get_conv2d_sequence(conv_args, circular_axis=1)
        self.pool = _GLOBAL_POOL[pool](axis=0)
        mlp_convs_args = []
        for in_ch, out_ch in zip(mlp_args[:-1], mlp_args[1:]):
            mlp_convs_args.append({
                'in_channels': in_ch,
                'out_channels': out_ch,
                'kernel_size': 1,
            })
        self.mlp, _, _ = get_conv1d_sequence(mlp_convs_args,
                                             output_act=output_act)

    def forward(self, x, weight=None, return_hidden=0):
        """
        Args:
            x : input data
            weight : weight for pooling
            return_hidden : int, return the last `return_hidden` hidden
                            layers output.
        """
        y = self.convs(x)
        if weight is None:
            y = self.pool(y)
        else:
            y = self.pool(y, weight=weight)
        if return_hidden > 0:
            y = self.mlp[:-return_hidden](y)
        else:
            y = self.mlp(y)
        return y


class UniformDoaFeatureNet(SerializableModule):
    """
    A mult-task network takes STFT as input (T-F),
    and extracts DOA-wise features (mapping to T-D)
    with convolutions and residual connections.
    After the features are mapped to DOA, sub-task branches process
    the data and output the multi-task predictions.

    The sub-task branches can be either frame-wise or sequence-wise.
        - Frame-wise branch simply consists of convolution along T-D.
        - Sequence-wise branch includes convolution, (weighted) temporal
          pooling, and MLP,
    """
    def __init__(self, n_freq, n_doa, ds_convs, n_rblk, tfd_factor, branches):
        """
        Args:
            n_freq : number of frequency bins
            n_doa  : number of directions-of-arrival
            ds_convs : downsampling layers at the beginning of the network,
                       list of arguments (in dict) for creating nn.Conv2d.
            n_rblk : number of residual blocks in the shared layers
            tfd_factor : size of features on each Time-Frequency-DOA point
            branches : definition of task-specific branches
                            list of tuple (`btype`, `args`, `weight_tid`)
                btype : 'fw' OR 'sw'
                args : arguments for _FrameWiseBranch or _SequenceWiseBranch
                weight_tid : ID of the other task which is used as weighting
                             input for this branch
            circular_doa : apply circular padding on convolutions along
                           the DOA axis.
        """
        super().__init__({
            'n_freq': n_freq,
            'n_doa': n_doa,
            'ds_convs': ds_convs,
            'n_rblk': n_rblk,
            'tfd_factor': tfd_factor,
            'branches': branches,
        })

        # trunk
        # down-sampling
        ds_seq, (_, ds_n_freq), ds_n_feat = get_conv2d_sequence(
            ds_convs,
            map_size=(0, n_freq),
        )
        # residual modules
        res_blocks = [
            ResidualBlock(ds_n_feat, ds_n_feat, padding_mode='replicate')
            for _ in range(n_rblk)
        ]
        # mapping to doa
        doa_mapper = _DoaMapper(ds_n_freq, n_doa, ds_n_feat, tfd_factor)
        # trunk module
        self.shared = nn.Sequential(ds_seq, *res_blocks, doa_mapper)

        # branches
        self.branches = nn.ModuleList()
        self.weight_tids = []
        for btype, args, weight_tid in branches:
            if btype.startswith('f'):
                # frame-wise pred. branch
                self.branches.append(_FrameWiseBranch(**args))
                self.weight_tids.append(None)
            else:
                assert btype.startswith('s')
                self.branches.append(_SequenceWiseBranch(**args))
                self.weight_tids.append(weight_tid)

    def forward(self, x, **kwargs):
        """
        Args:
            x : torch.Tensor of (samples, channels, time, frequency)
            branch_args : dict (branch id -> key word args for branch forward)

        Returns:
            list of task-specific predictions
        """
        doa_feat = self.shared(x)

        results = [None] * len(self.branches)
        while None in results:
            for i in range(len(self.branches)):
                if results[i] is not None:
                    continue
                wtid = self.weight_tids[i]
                prefix = f'b{i}_'
                args = {}
                for k, v in kwargs.items():
                    if k.startswith(prefix):
                        args[k[len(prefix):]] = v
                if wtid is None:
                    results[i] = self.branches[i](doa_feat, **args)
                elif results[wtid] is not None:
                    results[i] = self.branches[i](doa_feat,
                                                  weight=results[wtid],
                                                  **args)
        return [self._reformat(r) for r in results]

    def _reformat(self, y):
        # squeeze pred dimension
        return y.squeeze(1)
