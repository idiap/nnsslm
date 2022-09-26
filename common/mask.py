"""
mask.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import torch

import apkit


def apply_mask_blocks(fs, sig, mask_blocks, block_hop, stft_win, stft_hop,
                      min_freq, max_freq):
    assert block_hop % stft_hop == 0
    tf = apkit.stft(sig,
                    apkit.cola_hamming,
                    stft_win,
                    stft_hop,
                    last_sample=True)
    n_ch, n_frame, n_fbin = tf.shape
    mask = np.zeros((n_frame, n_fbin))
    weights = np.zeros(n_frame)

    n_blocks, n_frame_per_block, n_mask_fbin = mask_blocks.shape
    min_fbin = int(min_freq * stft_win // fs)
    max_fbin = int(max_freq * stft_win // fs)
    assert max_fbin - min_fbin == n_mask_fbin

    for bid in range(n_blocks):
        fid = bid * block_hop // stft_hop
        mask[fid:fid + n_frame_per_block,
             min_fbin:max_fbin] += mask_blocks[bid]
        mask[fid:fid + n_frame_per_block,
             n_fbin - min_fbin:n_fbin - max_fbin:-1] += mask_blocks[bid]
        weights[fid:fid + n_frame_per_block] += 1.0

    # there might be zero weight at the ending frames
    weights[weights == 0.0] += 1.0
    # and then normalize by weight
    mask /= weights[:, np.newaxis]

    # apply masking
    masked_tf = tf * mask
    # the masked signal is still a Fourier transform of a real signal
    assert np.allclose(masked_tf[:, :, 1:stft_win // 2],
                       masked_tf[:, :, :stft_win // 2:-1].conj())
    sig = apkit.istft(masked_tf, stft_hop)
    return sig


def minimum_norm_binary_mask(net, x, l_doa, forward_kargs={}):
    """
    Compute minimum-norm binary mask that produce greater or equal score
    at the target DOA.

    Args:
        net : network model
        x : input, torch.Tensor with axes
                   (frame, feature, map_axis_1, map_axis_2)
        l_doa : list of target DOA ids
        forward_kargs : model forward keyword arguments

    Returns:
        binary mask per frame : numpy array with axes (frame, time, freq)
        prediction of the masked input
    """
    assert x.dim() == 4
    m = torch.ones(x.size(0),
                   1,
                   x.size(2),
                   x.size(3),
                   device=x.device,
                   requires_grad=True)
    cont = True
    while cont:
        y = net(x * m, **forward_kargs)
        assert y.dim() == 2
        z = torch.sum(y[torch.arange(len(x)), l_doa])
        z.backward()

        n = m * (m.grad >= 0)

        if torch.allclose(n, m):
            cont = False
        else:
            m = n.detach().requires_grad_()

        cont = False  # one iteration only

    y = net(x * m, **forward_kargs)
    return m[:, 0], y


_STEP_SIZE = 0.01


def minimum_loss_mask(net, x, l_doa, loss_func, forward_kargs={}, alpha=0.0):
    """
    Estimate the soft-mask that produce minimum loss.

    Args:
        net : network model
        x : input, torch.Tensor with axes
                   (frame, feature, map_axis_1, map_axis_2)
        l_doa : list of target DOA ids
        loss_func(y, l_doa) : loss function between network output y and target doa
        forward_kargs : model forward keyword arguments
        alpha : weighting factor for L1-regularization on m

    Returns:
        binary mask per frame : numpy array with axes (frame, time, freq)
        prediction of the masked input
    """
    n_samples, n_feat, m1_size, m2_size = x.size()
    m = torch.ones(n_samples,
                   1,
                   m1_size,
                   m2_size,
                   device=x.device,
                   requires_grad=True)
    cont = True
    k = float('+inf')
    while cont:
        # forward
        y = net(x * m, **forward_kargs)
        assert y.dim() == 2

        # loss
        l = torch.sum(loss_func(y, l_doa)) \
            + alpha * torch.sum(m) / (m1_size * m2_size)
        print(f'avg loss : {l.data.item()/len(l_doa):.3g}')

        # gradient descent
        l.backward()
        n = m - torch.sign(m.grad) * _STEP_SIZE
        n.clamp_(0.0, 1.0)

        # stop criterion
        if l > k:
            cont = False
        else:
            k = l
            m = n.detach().requires_grad_()

    y = net(x * m, **forward_kargs)
    return m[:, 0], y
