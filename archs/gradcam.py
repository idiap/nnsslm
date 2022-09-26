"""
gradcam.py

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import numpy as np
import torch


class GradCamable(object):
  """
    Abstract model with a intermediate feature mapping representation.
    """
  def forward_feature_output(self, x, **kwargs):
    """
        Forward and get both feature activation maps and network output.

        Args:
            x : input data (torch.Tensor) with axes (samples, ...)
            kwargs : other arguments

        Returns:
            feature activation maps : torch.Tensor
                    with axes (samples, feature, map_axis_1, ..., map_axis_n)
            network output : torch.Tensor with axes (samples, ...)
        """
    raise NotImplementedError()


class StoreGrad(object):
  def __call__(self, grad):
    """
        Args:
            gradient to store   
        """
    self.grad = grad

  def get(self):
    """
        Returns:
            stored gradient    
        """
    try:
      return self.grad
    except AttributeError as e:
      raise RuntimeError(f'No gradient is stored ({e})')


def compute_gradcam(model: GradCamable, x, get_class_act, forward_kargs={}):
  """
    Compute Grad-CAM

    Args:
        x : input data (torch.Tensor) for model
        model : (GradCamable) network model
        get_class_act : function to the target class activation from the
                        network output. It has a signature of
                            get_class_act(torch.Tensor) -> torch.Tensor (scalar)
        forward_kargs : other arguments for forward computation

    Returns:
        cam : Grad-CAM of each input sample;
              numpy array with axes (sample, map_axis_1, ..., map_axis_n)
    """
  store_grad = StoreGrad()
  x.requires_grad_(True)
  feat_act, output = model.forward_feature_output(x, **forward_kargs)
  feat_act.register_hook(store_grad)
  class_act = get_class_act(output)
  torch.sum(class_act).backward()
  feat_grad = store_grad.get()

  # to numpy
  feat_act = feat_act.data.cpu().numpy()
  # feat_act (sample, feature, map_axis_1, ..., map_axis_n)
  feat_grad = feat_grad.data.cpu().numpy()
  # feat_grad (sample, feature, map_axis_1, ..., map_axis_n)
  assert feat_act.shape == feat_grad.shape
  map_dim = feat_act.ndim - 2
  alpha = np.mean(feat_grad, axis=tuple(range(2, map_dim + 2)), keepdims=True)
  # alpha (sample, feature, 1, ..., 1)
  cam = np.sum(feat_act * alpha, axis=1)
  # cam (sample, map_axis_1, ..., map_axis_n)
  assert cam.shape[0] == feat_act.shape[0]
  assert cam.shape[1:] == feat_act.shape[2:]
  return cam
