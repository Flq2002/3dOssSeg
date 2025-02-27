# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import weighted_loss
from torch.cuda.amp import autocast
from PIL import Image


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


class TextureL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, kernel_size=3, eps=1e-4, reduction='mean', loss_weight=1.0):
        super(TextureL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kernel_size = kernel_size
        self.sobel = SobelOperator(eps)
    # @autocast(True)
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred_texture = pred.sigmoid()
        pred_texture = self.sobel(pred_texture)
        target_texture = self.sobel(target.to(pred_texture.dtype))

        loss_texture = self.loss_weight * l1_loss(
            pred_texture, target_texture, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_texture


class SobelOperator(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        # Updated kernels for 3D
        x_kernel = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                             [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                             [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]) / 9
        self.conv_x = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor(x_kernel,device = 'cuda:0')[None,None,...].float()
        self.conv_x.weight.requires_grad = False

        y_kernel = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]) / 9
        self.conv_y = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight.data = torch.tensor(y_kernel,device = 'cuda:0')[None,None,...].float()
        self.conv_y.weight.requires_grad = False

        z_kernel = np.array([[[1, 2, 1], [1, 2, 1], [1, 2, 1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-1, -2, -1], [-1, -2, -1], [-1, -2, -1]]]) / 9
        self.conv_z = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_z.weight.data = torch.tensor(z_kernel,device = 'cuda:0')[None,None,...].float()
        self.conv_z.weight.requires_grad = False
    @autocast(True)
    def forward(self, x):
        # Expecting input shape (B, C, D, H, W)
        b, c, d, h, w = x.shape
        if c > 1:
            x = x.view(b*c, 1, d, h, w)
        
        x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        grad_z = self.conv_z(x)
        
        # Calculate gradient magnitude
        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + self.epsilon)

        x = x.view(b, c, d, h, w)

        return x