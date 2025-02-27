import torch
import torch.nn as nn
from torch.autograd import grad
from torch.cuda.amp import autocast
import torch.nn.functional as F
class CrossGradientLoss(nn.Module):

    def __init__(self, weight=1.):
        super().__init__()
        self.grad = SobelOperator()
    def forward(self, m1, m2, mask):
        """m1, m2: (b, c, x, y, z)"""
        
        # 计算每个方向的梯度
        grad1 = torch.stack(self.grad(m1), dim=-1)  # (b, c, x, y, z, 3)
        grad2 = torch.stack(self.grad(m2), dim=-1)  # (b, c, x, y, z, 3)

        # 计算梯度向量的点积
        dot_product = torch.sum(grad1 * grad2, dim=-1)  # (b, c, x, y, z)

        # 计算梯度向量的模
        norm1 = torch.sqrt(torch.sum(grad1 ** 2, dim=-1))  # (b, c, x, y, z)
        norm2 = torch.sqrt(torch.sum(grad2 ** 2, dim=-1))  # (b, c, x, y, z)

        # 计算余弦相似度
        cosine_similarity = dot_product / (norm1 * norm2 + 1e-6)  # 加入一个小常数避免除零错误

        # 只计算mask=1的部分
        cosine_similarity = cosine_similarity * mask  # 通过mask筛选出需要计算的位置
        # 计算最终的损失，返回平均的余弦相似度损失
        loss = torch.mean(cosine_similarity)
        # print("loss",loss)
        return 1-loss
import numpy as np
class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-6):
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
    # @autocast(True)
    def forward(self, x):
        # Expecting input shape (B, C, D, H, W)
        b, c, d, h, w = x.shape
        if c > 1:
            x = x.view(b*c, 1, d, h, w)
        
        x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        grad_x = self.conv_x(x).view(b, c, d, h, w)
        grad_y = self.conv_y(x).view(b, c, d, h, w)
        grad_z = self.conv_z(x).view(b, c, d, h, w)

        return (grad_x,grad_y,grad_z)