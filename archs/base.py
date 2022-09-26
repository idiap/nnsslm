"""
base.py

Base classes

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import pickle
import itertools

import torch
import torch.nn as nn

import numpy as np

_MODEL_SUFFIX = '.model.pkl'
_ARCH_SUFFIX = '.arch.pkl'

# pytorch convolution layer arguments
CONV_ARG_IN_CHS = 'in_channels'
CONV_ARG_OUT_CHS = 'out_channels'
CONV_ARG_KERNEL = 'kernel_size'
CONV_ARG_STRIDE = 'stride'
CONV_ARG_PAD = 'padding'
CONV_ARG_DIAL = 'dialation'


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class SigmoidNoSaturation(nn.Module):
    """
    Sigmoid function bewtween -0.05 and 1.05, so that it doesn't saturate
    between 0 and 1
    """
    def forward(self, x):
        return -0.05 + torch.sigmoid(x) * 1.1


class SigmoidScale10(nn.Module):
    """
    Sigmoid function scaling input by 1/10, so that it doesn't saturate easily
    """
    def forward(self, x):
        return torch.sigmoid(x / 10.0)


# activation function types
ACT_NONE = 0
ACT_SIGMOID = 1
ACT_TANH = 2
ACT_RELU = 3
ACT_SIG_NOSAT = 4
ACT_SIG_10 = 5

# instructions
ACT_INSTRUCTION = '{0:None, 1:Sigmoid, 2:Tanh, 3:ReLU,' \
                  ' 4:Sigmoid(no saturation), 5:Sigmoid(x/100)}'

# activation functions
_act_funcs = [
    Identity,
    nn.Sigmoid,
    nn.Tanh,
    nn.ReLU,
    SigmoidNoSaturation,
    SigmoidScale10,
]


def num_params(net):
    return np.sum([np.prod(x.size()) for x in net.parameters()])


class SerializableModule(nn.Module):
    """Serializable (model + architecture) NN module
    """
    def __init__(self, args):
        super().__init__()
        self.arch_args = args

    def save(self, name_prefix):
        torch.save(self.state_dict(), name_prefix + _MODEL_SUFFIX)
        arch = (self.__class__, self.arch_args)
        with open(name_prefix + _ARCH_SUFFIX, 'wb') as f:
            pickle.dump(arch, f)


def load_module(name_prefix):
    with open(name_prefix + _ARCH_SUFFIX, 'rb') as f:
        arch_class, arch_args = pickle.load(f)
    net = arch_class(**arch_args)
    net.load_state_dict(torch.load(name_prefix + _MODEL_SUFFIX))
    return net


def _conv3x3(in_channels, out_channels, stride=1):
    # 3x3 Convolution
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample=None,
                 padding_mode='zeros'):
        super().__init__()
        seq = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      padding_mode=padding_mode,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.mseq = nn.Sequential(*seq)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.mseq(x)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


def copy_model(dst, src):
    """
    Copy model parameters (and buffers).
    Destimation model parameters should be same or a subset of the source
    model parameters.

    Args:
        dst : destination model
        src : destination model
    """
    # copy parameters
    with torch.no_grad():
        src_params = dict(src.named_parameters())
        for name, param in dst.named_parameters():
            param.data.copy_(src_params[name].data)

    # copy buffer
    with torch.no_grad():
        src_buf = dict(src.named_buffers())
        for name, buf in dst.named_buffers():
            buf.copy_(src_buf[name])


def is_module_gpu(module):
    """
    Returns:
        boolean value indicate if module is in GPU
    """
    return next(module.parameters()).data.is_cuda


class WeightInitializer(object):
    def __init__(self, init_func=nn.init.xavier_uniform_, **kwargs):
        """
        Args:
            init_func : see torch.nn.init
            kwargs : arguments forward to init_func
        """
        self.init_func = init_func
        self.kwargs = kwargs

    @torch.no_grad()
    def __call__(self, module):
        module.apply(self.init_weight)

    @torch.no_grad()
    def init_weight(self, module):
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            self.init_func(module.weight, **self.kwargs)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class CircularPadding2d(nn.Module):
    """
    Circular padding
    (nothing to explain I think)
    """
    def __init__(self, padding):
        """
        Args:
            padding : tuple or list, padding size on both sides of the last
                      dimensions.
                      or int for 1-dim padding.
                      NOTE: the order is same as nn.Conv2d, but reverse of
                      nn.ConstantPad2d.
        """
        super().__init__()
        self.padding = padding
        if type(padding) == int:
            self.padding_reorg = list(itertools.repeat(padding, 4))
        else:
            assert len(padding) == 2
            self.padding_reorg = [
                p for q in self.padding[::-1] for p in itertools.repeat(q, 2)
            ]

    def forward(self, x):
        return nn.functional.pad(x, self.padding_reorg, 'circular')

    def extra_repr(self):
        return str(self.padding)


def get_conv1d_sequence(conv_args,
                        hidden_act=ACT_RELU,
                        output_act=None,
                        batch_norm=True,
                        map_size=0):
    """
    Construct a stack of 1D convolution layers, with batch_norm and
    activitaion functions between convolutions, and optionally another
    activation function at output.

    Args:
        conv_args : list of nn.Conv1d arguments
        hidden_act : activation function between hidden layers
        output_act : activation function at output.
                     if None, hidden activation and batch-norm (if applicable)
                     are added to the last layer.
        batch_norm : add batch-normalization after hidden activation
        map_size : input map size, int
                   The output map size is computed according to this.
                   However, if output map size is not needed,
                   input map size can be just any number (for example 0).

    Returns:
        module : nn.Module
        out_map_size : output_map_size
        out_channels : number of output channels
    """
    seq = []
    n_ch = None
    for i, l in enumerate(conv_args):
        # check consistency
        assert CONV_ARG_DIAL not in l
        assert n_ch is None or l[CONV_ARG_IN_CHS] == n_ch
        n_ch = l[CONV_ARG_OUT_CHS]

        # compute map size
        if CONV_ARG_KERNEL in l:
            kernel = l[CONV_ARG_KERNEL]
        else:
            kernel = 1
        if CONV_ARG_PAD in l:
            padding = l[CONV_ARG_PAD]
        else:
            padding = 0
        if CONV_ARG_STRIDE in l:
            stride = l[CONV_ARG_STRIDE]
        else:
            stride = 1
        map_size = (map_size + 2 * padding - kernel + stride) // stride

        seq.append(nn.Conv1d(**l))
        if i == len(conv_args) - 1 and output_act is not None:
            # last layer is consider output layer, and use output_act
            seq.append(_act_funcs[output_act]())
        else:
            # hidden layer
            if batch_norm:
                seq.append(nn.BatchNorm1d(n_ch))
            seq.append(_act_funcs[hidden_act]())

    return nn.Sequential(*seq), map_size, n_ch


def get_conv2d_sequence(conv_args,
                        hidden_act=ACT_RELU,
                        output_act=None,
                        batch_norm=True,
                        circular_axis=None,
                        map_size=(0, 0)):
    """
    Construct a stack of 2D convolution layers, with batch_norm and
    activitaion functions between convolutions, and optionally another
    activation function at output.

    Args:
        conv_args : list of nn.Conv2d arguments
        hidden_act : activation function between hidden layers
        output_act : activation function at output.
                     if None, hidden activation and batch-norm (if applicable)
                     are added to the last layer.
        batch_norm : add batch-normalization after hidden activation
        circular_axis : axis for circular padding
        map_size : input map size, tuple of (height, width).
                   The output map size is computed according to this.
                   However, if output map size (at any axis) is not needed,
                   input map size can be just any number (for example 0).

    Returns:
        module : nn.Module
        out_map_size : output_map_size
        out_channels : number of output channels
    """
    seq = []
    n_ch = None
    height, width = map_size
    for i, l in enumerate(conv_args):
        # check consistency
        assert CONV_ARG_DIAL not in l
        assert n_ch is None or l[CONV_ARG_IN_CHS] == n_ch
        n_ch = l[CONV_ARG_OUT_CHS]

        # compute map size
        if CONV_ARG_KERNEL in l:
            if isinstance(l[CONV_ARG_KERNEL], int):
                kernel_height = l[CONV_ARG_KERNEL]
                kernel_width = l[CONV_ARG_KERNEL]
            else:
                kernel_height, kernel_width = l[CONV_ARG_KERNEL]
        else:
            kernel_height = 1
            kernel_width = 1
        if CONV_ARG_PAD in l:
            if isinstance(l[CONV_ARG_PAD], int):
                padding_height = l[CONV_ARG_PAD]
                padding_width = l[CONV_ARG_PAD]
            else:
                padding_height, padding_width = l[CONV_ARG_PAD]
        else:
            padding_height = 0
            padding_width = 0
        if CONV_ARG_STRIDE in l:
            if isinstance(l[CONV_ARG_STRIDE], int):
                stride_height = l[CONV_ARG_STRIDE]
                stride_width = l[CONV_ARG_STRIDE]
            else:
                stride_height, stride_width = l[CONV_ARG_STRIDE]
        else:
            stride_height = 1
            stride_width = 1
        height = (height + 2 * padding_height - kernel_height +
                  stride_height) // stride_height
        width = (width + 2 * padding_width - kernel_width +
                 stride_width) // stride_width

        if circular_axis is None:
            # nothing special
            seq.append(nn.Conv2d(**l))
        elif circular_axis == 0:
            # apply circular padding on axis 0
            # modify arguments by remove padding on axis 0
            l_mod = l.copy()
            l_mod[CONV_ARG_PAD] = (0, padding_width)
            # add circular padding
            if padding_height > 0:
                seq.append(CircularPadding2d((padding_height, 0)))
            # add convolution
            seq.append(nn.Conv2d(**l_mod))
        elif circular_axis == 1:
            # apply circular padding on axis 0
            # modify arguments by remove padding on axis 0
            l_mod = l.copy()
            l_mod[CONV_ARG_PAD] = (padding_height, 0)
            # add circular padding
            if padding_width > 0:
                seq.append(CircularPadding2d((0, padding_width)))
            # add convolution
            seq.append(nn.Conv2d(**l_mod))
        else:
            raise ValueError(
                f'circular_axis must be 0 or 1 (given {circular_axis})')
        if i == len(conv_args) - 1 and output_act is not None:
            # last layer is consider output layer, and use output_act
            seq.append(_act_funcs[output_act]())
        else:
            # hidden layer
            if batch_norm:
                seq.append(nn.BatchNorm2d(n_ch))
            seq.append(_act_funcs[hidden_act]())

    return nn.Sequential(*seq), (height, width), n_ch


class GlobalPooling(object):
    """
    Global pooling of any reduce function
    """
    def __init__(self, reduce_func):
        self.reduce_func = reduce_func

    def __call__(self, axis):
        def _pool(x):
            return self.reduce_func(x, dim=2 + axis)

        return _pool


class GlobalStatisticPooling(object):
    """
    Global statistic (mean + standard deviation)  pooling
    """
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x):
        mean = torch.mean(x, dim=2 + self.axis)
        std = torch.std(x, dim=2 + self.axis, unbiased=False)
        return torch.cat([mean, std], dim=1)

_EPSILON = 1e-6

class GlobalWeightedAveragePooling(object):
    """
    Global statistic (mean + standard deviation)  pooling
    """
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x, weight):
        xsum = torch.sum(x * weight, dim=2 + self.axis)
        wsum = torch.sum(weight, dim=2 + self.axis) + _EPSILON
        p = xsum / wsum
        return p


class GlobalWeightedStatisticPooling(object):
    """
    Global statistic (mean + standard deviation)  pooling
    """
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x, weight):
        aid = 2 + self.axis
        xsum = torch.sum(x * weight, dim=aid, keepdim=True)
        wsum = torch.sum(weight, dim=aid, keepdim=True) + _EPSILON
        wmean = xsum / wsum
        sqsum = torch.sum(((x - wmean)**2.0) * weight, dim=aid, keepdim=True)
        wstd = torch.sqrt(sqsum / wsum + _EPSILON)
        p = torch.cat([wmean, wstd], dim=1).squeeze(aid)
        return p


def _torch_max_no_index(x, **kwargs):
    return torch.max(x, **kwargs)[0]


# pooling methods
POOL_NONE = 0
POOL_MAX = 1
POOL_AVG = 2
POOL_STAT = 3
POOL_WEIGHTED_AVG = 4
POOL_WEIGHTED_STAT = 5

_GLOBAL_POOL = [
    Identity,
    GlobalPooling(_torch_max_no_index),
    GlobalPooling(torch.mean),
    GlobalStatisticPooling,
    GlobalWeightedAveragePooling,
    GlobalWeightedStatisticPooling,
    GlobalWeightedStatisticPooling,
]
