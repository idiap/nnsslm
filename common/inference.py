"""
inference.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import torch

import archs

from .mask import minimum_norm_binary_mask, minimum_loss_mask


def extract_framewise_feature(fs,
                              sig,
                              win_size,
                              hop_size,
                              extract_ft,
                              dtype='float32',
                              ctx_size=0,
                              ahd_size=0):
    """
    Apply framing to audio signal and extract features per frame.

    Args:
        fs : sampling rate
        sig : time-domain signal
        win_size : window size
        hop_size : hop size
        extract_ft : function to extract features (see .features)
        dtype : output data type

    Returns:
        frame-wise feature : numpy array with axes (time, feature)
    """
    if ctx_size > 0 or ahd_size > 0:
        raise NotImplementedError('see ssl_nn_v2/test_nn_raw.py')

    return np.array(
        [
            extract_ft(fs, sig[:, o:o + win_size]) for o in range(
                0,
                sig.shape[1] - win_size + 1,
                hop_size,
            )
        ],
        dtype=dtype,
    )


def predict_batch(net, x, batch=100, gpu=True, forward_kargs={}):
    """
    Make prediction in mini-batches

    Args:
        net : network model
        x : input feature, numpy array with axes (frame, feature, ...)
        batch : mini-batch size
        gpu : use gpu
        forward_kargs : model forward keyword arguments

    Returns:
        if multi-task:
            list of predictions, one numpy array per task
        otherwise:
            numpy array of prediction (frame, prediction, ...)
    """
    assert len(x) > 0

    result = None
    with torch.no_grad():
        for i in range(0, len(x), batch):
            j = min(i + batch, len(x))
            y = torch.from_numpy(x[i:j])
            if gpu:
                y = y.cuda()
            output = net(y, **forward_kargs)
            if type(output) is list:
                if result is None:
                    n_task = len(output)
                    mt = True
                    result = [[] for _ in range(n_task)]
                for t in range(n_task):
                    result[t].append(output[t].cpu().data.numpy())
            else:
                if result is None:
                    mt = False
                    result = []
                result.append(output.cpu().data.numpy())
    if mt:
        return [np.concatenate(r) for r in result]
    else:
        return np.concatenate(result)


class GetByClassID(object):
    """
    Get class activation by index
    """
    def __init__(self, indices, gpu=False):
        self.indices = torch.LongTensor(indices)
        self.gpu = gpu

        if gpu:
            self.indices = self.indices.cuda()

    def __call__(self, y):
        assert y.dim() == 2
        assert y.size(0) == self.indices.size(0)
        sids = torch.arange(y.size(0))
        if self.gpu:
            sids = sids.cuda()
        return y[sids, self.indices]


def compute_gradcam_batch(net,
                          x,
                          get_class_act,
                          batch=100,
                          gpu=True,
                          forward_kargs={}):
    """
    Compute grad-cam in batches.

    Args:
        net : network model
        x : input feature, numpy array with axes (frame, feature(s)...)
        get_class_act : function to get class activation
                        (see archs.compute_gradcam())
                        or list of target class id
        batch : mini-batch size
        gpu : use gpu
        forward_kargs : model forward keyword arguments

    Returns:
        Grad-CAM per frame : numpy array with axes (frame, time, freq)
    """
    if type(get_class_act) == list:
        target_class_ids = get_class_act
    else:
        target_class_ids = None
    result = []
    for i in range(0, len(x), batch):
        j = min(i + batch, len(x))
        y = torch.from_numpy(x[i:j])
        if gpu:
            y = y.cuda()
        if target_class_ids is not None:
            gca = GetByClassID(target_class_ids[i:j], gpu)
        else:
            gca = get_class_act
        cam = archs.compute_gradcam(net, y, gca, forward_kargs=forward_kargs)
        result.append(cam)
    return np.concatenate(result)


@torch.no_grad()
def compute_tfpact_batch(net,
                         x,
                         batch=100,
                         gpu=True,
                         forward_kargs={},
                         l_doa=None):
    """
    Compute time-frequency local prediction activation in batches.

    Args:
        net : network model
        x : input feature, numpy array with axes (frame, feature(s)...)
        batch : mini-batch size
        gpu : use gpu
        forward_kargs : model forward keyword arguments
        l_doa : list of target DOA ids

    Returns:
        TFP act. per frame : numpy array with axes (frame, time, freq)
    """
    result = []
    for i in range(0, len(x), batch):
        j = min(i + batch, len(x))
        y = torch.from_numpy(x[i:j])
        if gpu:
            y = y.cuda()
        tfp, o = net.forward_feature_output(y, **forward_kargs)
        if l_doa is None:
            _, l_doa_batch = torch.max(o, dim=1)
            l_doa_batch = l_doa_batch.data.cpu().numpy()
        else:
            l_doa_batch = l_doa[i:j]
        tfp = tfp.data.cpu().numpy()
        cam = tfp[range(len(tfp)), l_doa_batch]
        result.append(cam)
    return np.concatenate(result)


def compute_minnm_batch(net, x, l_doa, batch=100, gpu=True, forward_kargs={}):
    """
    Compute minimum-norm binary mask that produce greater or equal score
    at the target DOA.

    Args:
        net : network model
        x : input feature, numpy array with axes
                           (frame, feature, map_axis_1, map_axis_2)
        l_doa : list of target DOA ids
        batch : mini-batch size
        gpu : use gpu
        forward_kargs : model forward keyword arguments

    Returns:
        binary mask per frame : numpy array with axes (frame, time, freq)
        predction of the masked input
    """
    l_cam = []
    l_npred = []
    for i in range(0, len(x), batch):
        j = min(i + batch, len(x))
        y = torch.from_numpy(x[i:j])
        if gpu:
            y = y.cuda()
        cam, y = minimum_norm_binary_mask(net, y, l_doa[i:j], forward_kargs={})
        l_cam.append(cam.data.cpu().numpy())
        l_npred.append(y.data.cpu().numpy())
    return np.concatenate(l_cam), np.concatenate(l_npred)


def compute_minlm_batch(net,
                        x,
                        l_doa,
                        loss_func,
                        batch=100,
                        gpu=True,
                        forward_kargs={},
                        alpha=0.0):
    """
    Estimate the soft-mask that produce minimum loss in batches.

    Args:
        net : network model
        x : input feature, numpy array with axes
                           (frame, feature, map_axis_1, map_axis_2)
        l_doa : list of target DOA ids
        loss_func(y, l_doa) : loss function between network output y and target doa
        batch : mini-batch size
        gpu : use gpu
        forward_kargs : model forward keyword arguments
        alpha : weighting factor for L1-regularization on m

    Returns:
        mask per frame : numpy array with axes (frame, time, freq)
        predction of the masked input
    """
    l_cam = []
    l_npred = []
    for i in range(0, len(x), batch):
        j = min(i + batch, len(x))
        y = torch.from_numpy(x[i:j])
        if gpu:
            y = y.cuda()
        cam, y = minimum_loss_mask(net,
                                   y,
                                   l_doa[i:j],
                                   loss_func,
                                   forward_kargs={},
                                   alpha=alpha)
        l_cam.append(cam.data.cpu().numpy())
        l_npred.append(y.data.cpu().numpy())
    return np.concatenate(l_cam), np.concatenate(l_npred)
