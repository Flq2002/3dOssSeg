# -----------------------------------------------------------------------------------
# References:
# https://github.com/MengyuWang826/SegRefiner/blob/main/mmdet/models/detectors/segrefiner_base.py
# -----------------------------------------------------------------------------------

# only for binary segmentation

from collections import OrderedDict
import torch
import torch.distributed as dist
import numpy as np
from mmcv.runner import BaseModule
from Segrefiner.guided_diffusion.DenoiseUNet_gaussian import DenoiseUNet
from Segrefiner.losses.cross_entropy_loss import CrossEntropyLoss
from Segrefiner.losses.textrue_l1_loss import TextureL1Loss
import torch.nn.functional as F
from utils.nms3d import aligned_3d_nms
from monai.losses import DiceCELoss
from Segrefiner.losses.region_loss import *
from Segrefiner.segrefiner_base import SegRefiner
from Segrefiner.segrefiner_base import _get_patch_coors
from data.gen_patches import get_dets

def sliding_window_coordinates(P, window_size, overlap):
    step = int(window_size * (1 - overlap))
    
    # 生成起始坐标
    x_starts = np.arange(0, P[1] - window_size + 1, step)
    y_starts = np.arange(0, P[0] - window_size + 1, step)
    z_starts = np.arange(0, P[2] - window_size + 1, step)
    
    # 生成所有组合
    x1, y1, z1 = np.meshgrid(x_starts, y_starts, z_starts, indexing='ij')
    x2, y2, z2 = x1 + window_size, y1 + window_size, z1 + window_size

    # 合并坐标
    coords = np.stack((x1, y1, z1, x2, y2, z2), axis=-1).reshape(-1, 6)

    window_size = window_size//2
    center = np.array([[P[1]//2-window_size, P[0]//2-window_size,P[2]//2-window_size,
              P[1]//2+window_size, P[0]//2+window_size, P[2]//2+window_size]])
    coords = np.concatenate((coords,center),axis=0)
    return coords


from DiffVNet.guided_diffusion.gaussian_diffusion import GaussianDiffusion
from DiffVNet.guided_diffusion.gaussian_diffusion import get_named_beta_schedule,ModelMeanType, ModelVarType,LossType
class SegRefiner_oss_gaussian(SegRefiner):
    def __init__(self,
                 diffusion_cfg,
                 denoise_cfg,):
        super(SegRefiner_oss_gaussian,self).__init__(diffusion_cfg,denoise_cfg)
        # from IPython import embed;embed()
        self.denoise_model = DenoiseUNet(**denoise_cfg)
        self.gaussian = GaussianDiffusion(
            betas=get_named_beta_schedule("linear",self.num_timesteps),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE
        )
    def get_local_input(self, img, ori_size_mask, model_size):
        # print(ori_size_mask.shape)
        ori_size_fine_probs = torch.zeros_like(ori_size_mask)
        # patch_coors = get_dets(ori_size_mask,model_size,0.3)
        patch_coors = sliding_window_coordinates(ori_size_mask.shape[-3:],model_size[-1],0.5)
        return self.crop_patch(img, ori_size_mask, ori_size_fine_probs, patch_coors)
    
    
    def single_test(self, img, coarse_label, patch_size) -> torch.tensor :
        local_indices = list(range(self.num_timesteps))[::-1]
        # model size -> orig size
        ori_size_mask = coarse_label
        ori_size_mask = (ori_size_mask >= 0.5).float()

        patch_imgs, patch_masks, _ , patch_coors = \
        self.get_local_input(img, ori_size_mask, patch_size)
        # print(len(patch_imgs),ori_size_mask.shape)
        if patch_imgs is None:
            return ori_size_mask
        batch_max = 8
        num_ins = len(patch_imgs)
        if num_ins <= batch_max:
            xs = [(patch_masks, patch_imgs, None)]
        else:
            xs = []
            for idx in range(0, num_ins, batch_max):
                end = min(num_ins, idx + batch_max)
                xs.append((patch_masks[idx: end], patch_imgs[idx:end], None))
        # local_masks, _ = self.p_sample_loop(xs, 
        #                                     local_indices, 
        #                                     patch_imgs.device,
        #                                     use_last_step=True)
        local_masks = self.gaussian.ddim_sample_loop(self.denoise_model,
                                                     patch_imgs.shape,
                                                     noise=patch_masks,
                                                     model_kwargs={"img":patch_imgs},
                                                     clip_denoised=False)
        # from IPython import embed;embed()
        mask = self.paste_local_patch(local_masks["pred_xstart"].sigmoid(), ori_size_mask, patch_coors)
        return mask
        
    def forward_train(self, img, target, x_last) -> dict:
        current_device = img.device
        t = self.uniform_sampler(self.num_timesteps, img.shape[0], current_device)
        x_t = self.gaussian.q_sample(target, t, noise=x_last)
        # z_t = torch.cat((img, x_t), dim=1)
        pred_logits = self.denoise_model(x_t,t,img) 
        # from IPython import embed;embed()
        iou_pred = self.cal_iou(target, pred_logits.sigmoid())
        dice_pred = self.cal_dice(target, pred_logits.sigmoid())
        losses = dict()
        # region_weight = get_region_weight(pred_logits,target,x_last,num_regions=16)
        
        # losses['loss_region_dice'] = region_diceLoss(pred_logits,target,region_weight,num_regions=16)
        # losses['loss_mask'] = self.loss_CE(pred_logits, target)
        losses['loss_mask'] = self.loss_mask(pred_logits, target)
        losses['loss_texture'] = self.loss_texture(pred_logits, target)
        # from IPython import embed;embed()
        losses['iou'] = iou_pred.mean()
        losses['dice'] = dice_pred.mean()
        return losses