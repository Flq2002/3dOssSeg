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
from Segrefiner.guided_diffusion.DenoiseUNet import DenoiseUNet
from Segrefiner.losses.cross_entropy_loss import CrossEntropyLoss
from Segrefiner.losses.textrue_l1_loss import TextureL1Loss
import torch.nn.functional as F
from utils.nms3d import aligned_3d_nms
from monai.losses import DiceCELoss
# from Segrefiner.losses.region_loss import *

class SegRefiner(BaseModule):
    """Base class for detectors."""
    def __init__(self,
                 diffusion_cfg,
                 denoise_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SegRefiner, self).__init__(init_cfg)
        self.denoise_model = DenoiseUNet(**denoise_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_cfg)
        #NOTE loss only for binary seg 
        # self.loss_CE = CrossEntropyLoss(use_sigmoid=True)
        self.loss_mask = DiceCELoss(
            sigmoid = True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
        )
        self.loss_texture = TextureL1Loss()
        # self.loss_cross_gradient = CrossGradientLoss()
    def uniform_sampler(self,num_steps, batch_size, device):
        all_indices = np.arange(num_steps)
        indices_np = np.random.choice(all_indices, size=(batch_size,))
        indices = torch.from_numpy(indices_np).long().to(device)
        return indices  
    
    def _diffusion_init(self, diffusion_cfg):
        self.diff_iter = diffusion_cfg['diff_iter']
        betas = diffusion_cfg['betas']
        self.eps = 1.e-6
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]

    # def forward_train(self, img, target, x_last, x_t, t) -> dict:
    #     current_device = img.device
    #     # t = self.uniform_sampler(self.num_timesteps, img.shape[0], current_device)
    #     # x_t = self.q_sample(target, x_last, t, current_device)
    #     t = t.to(current_device)
    #     x_t = x_t.unsqueeze(1).to(current_device)
    #     # from IPython import embed;embed()
    #     z_t = torch.cat((img, x_t), dim=1)
    #     # z_t = torch.cat((img,img,img,x_t), dim=1)
    #     pred_logits = self.denoise_model(z_t, t) 
    #     iou_pred = self.cal_iou(target, pred_logits.sigmoid())
    #     dice_pred = self.cal_dice(target, pred_logits.sigmoid())
    #     losses = dict()
    #     # region_weight = get_region_weight(pred_logits,target,x_last,num_regions=16)
    #     # losses['loss_region_dice'] = region_diceLoss(pred_logits,target,region_weight,num_regions=16)
    #     # losses['loss_mask'] = self.loss_CE(pred_logits, target)

    #     # losses['loss_gradient'] = self.loss_cross_gradient(pred_logits,img,target)
        
    #     losses["loss_cover"] = F.relu(x_t - pred_logits.sigmoid()).sum()
    #     losses['loss_mask'] = self.loss_mask(pred_logits, target)
    #     losses['loss_texture'] = self.loss_texture(pred_logits, target)
    #     # from IPython import embed;embed()
    #     losses['iou'] = iou_pred.mean()
    #     losses['dice'] = dice_pred.mean()
    #     return losses, pred_logits.sigmoid()

    def forward_train(self, img, target, x_last) -> dict:
        current_device = img.device
        t = self.uniform_sampler(self.num_timesteps, img.shape[0], current_device)
        x_t = self.q_sample(target, x_last, t, current_device)
        z_t = torch.cat((img, x_t), dim=1)
        # z_t = torch.cat((img,img,img,x_t), dim=1)
        pred_logits = self.denoise_model(z_t, t)
        iou_pred = self.cal_iou(target, pred_logits.sigmoid())
        dice_pred = self.cal_dice(target, pred_logits.sigmoid())
        losses = dict()
        # region_weight = get_region_weight(pred_logits,target,x_last,num_regions=16)
        # losses['loss_region_dice'] = region_diceLoss(pred_logits,target,region_weight,num_regions=16)
        # losses['loss_mask'] = self.loss_CE(pred_logits, target)

        # losses['loss_gradient'] = self.loss_cross_gradient(pred_logits,img,target)
        
        losses["loss_cover"] = F.relu(x_t - pred_logits.sigmoid()).sum()
        losses['loss_mask'] = self.loss_mask(pred_logits, target)
        losses['loss_texture'] = self.loss_texture(pred_logits, target)
        # from IPython import embed;embed()
        losses['iou'] = iou_pred.mean()
        losses['dice'] = dice_pred.mean()
        return losses, pred_logits.sigmoid()
    
    def forward_train_ou(self, img, target, x_last) -> dict:
        current_device = img.device
        t = self.uniform_sampler(self.num_timesteps, img.shape[0], current_device)
        x_t = self.q_sample(target, x_last, t, current_device)
        z_t = torch.cat((img, x_t), dim=1)
        # z_t = torch.cat((img,img,img,x_t), dim=1)
        pred_logits = self.denoise_model(z_t, t)
        iou_pred = self.cal_iou(target, pred_logits.sigmoid())
        dice_pred = self.cal_dice(target, pred_logits.sigmoid())
        losses = dict()
        # region_weight = get_region_weight(pred_logits,target,x_last,num_regions=16)
        # losses['loss_region_dice'] = region_diceLoss(pred_logits,target,region_weight,num_regions=16)
        # losses['loss_mask'] = self.loss_CE(pred_logits, target)

        # losses['loss_gradient'] = self.loss_cross_gradient(pred_logits,img,target)
        
        losses["loss_cover"] = F.relu(pred_logits.sigmoid()-x_t).sum()
        losses['loss_mask'] = self.loss_mask(pred_logits, target)
        losses['loss_texture'] = self.loss_texture(pred_logits, target)
        # from IPython import embed;embed()
        losses['iou'] = iou_pred.mean()
        losses['dice'] = dice_pred.mean()
        return losses, pred_logits.sigmoid()
    @torch.no_grad()
    def cal_iou(self, target, mask, eps=1e-3):
        target = target.clone().detach() >= 0.5
        mask = mask.clone().detach() >= 0.5
        si = (target & mask).sum(-1).sum(-1).sum(-1)
        su = (target | mask).sum(-1).sum(-1).sum(-1)
        return (si / (su + eps))
    @torch.no_grad()
    def cal_dice(self, target, mask, eps=1e-3):
        target = target.clone().detach() >= 0.5
        mask = mask.clone().detach() >= 0.5
        si = (target & mask).sum(-1).sum(-1).sum(-1)
        return (2 * si) / (target.sum(-1).sum(-1).sum(-1) + mask.sum(-1).sum(-1).sum(-1) + eps)
    def _bitmapmasks_to_tensor(self, bitmapmasks, current_device):
        tensor_masks = []
        for bitmapmask in bitmapmasks:
            tensor_masks.append(bitmapmask.masks)
        tensor_masks = np.stack(tensor_masks)
        tensor_masks = torch.tensor(tensor_masks, device=current_device, dtype=torch.float32)
        return tensor_masks
    
    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        # from IPython import embed;embed()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample
    
    def p_sample_loop(self, xs, indices, current_device, use_last_step=True, save_diff = False):
        res, fine_probs = [], []
        for data in xs:
            x_last, img, cur_fine_probs = data # x_last: [B,C,H,W,D]
            if cur_fine_probs is None:
                cur_fine_probs = torch.zeros_like(x_last)
            x = x_last
            diff_x = []
            for i in indices:
                # if len(indices) == 1:
                # from IPython import embed;embed()
                t = torch.tensor([i] * x.shape[0], device=current_device)
                last_step_flag = (use_last_step and i==indices[-1])
                model_input = torch.cat((img, x), dim=1)
                # model_input = torch.cat((img,img,img, x), dim=1)
                x, cur_fine_probs = self.p_sample(model_input, cur_fine_probs, t)

                if last_step_flag:
                    x =  x.sigmoid()
                else:
                    sample_noise = torch.rand(size=x.shape, device=x.device)
                    fine_map = (sample_noise < cur_fine_probs).float()
                    pred_x_start = (x >= 0).float()
                    x = pred_x_start * fine_map + x_last * (1 - fine_map)
                    if save_diff:
                        diff_x.append(x.squeeze().cpu().data.numpy())
                    # from IPython import embed;embed()
                    # import SimpleITK as sitk
                    # test_save_path = '/media/HDD/fanlinqian/work_dirs_refiner/Exp_refiner_OSS/20241107-1730971940-diffusion_refiner_oss/res'
                    # savex = x.squeeze().cpu().data.numpy()
                    # out = sitk.GetImageFromArray((savex>=0.5).astype(np.float32))
                    # sitk.WriteImage(out, f'{test_save_path}/{i}.nii.gz')
            res.append(x)
            fine_probs.append(cur_fine_probs)
        res = torch.cat(res, dim=0)
        fine_probs = torch.cat(fine_probs, dim=0)
        if save_diff:
            return res, fine_probs, diff_x
        return res, fine_probs

    def p_sample(self, model_input, cur_fine_probs, t):
        """
        return 
            pred_logits: t时刻的mask
            cur_fine_probs: 置信度
        """
        pred_logits = self.denoise_model(model_input, t, in_test=True)
        t = t[0].item()
        x_start_fine_probs = 2 * torch.abs(pred_logits.sigmoid() - 0.5)
        # 用于度量模型的不确定程度，越接近0.说明模型预测出来的结果越不确定，对应论文中的p_theta(m_{0,t})
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        return pred_logits, cur_fine_probs
    @torch.no_grad()
    def single_test(self, img, coarse_label, patch_size, need_local_step=False, use_last_step=True) -> torch.tensor :
        ori_shape = img.shape[-3:]
        current_device = img.device
        indices = list(range(self.num_timesteps))[::-1]
        if need_local_step:
            global_indices = indices[:-1]
            local_indices = [indices[-1]]
        else:
            global_indices = indices
        # orig size -> model size
        global_img, global_mask = self.get_global_input(img, coarse_label, patch_size, current_device)
        model_size_mask, fine_probs = self.p_sample_loop([(global_mask, global_img, None)], 
                                                        global_indices, 
                                                        current_device, 
                                                        use_last_step=use_last_step,
                                                        )
        if need_local_step:
            # model size -> orig size
            ori_size_mask = F.interpolate(model_size_mask, size=ori_shape)
            ori_size_mask = (ori_size_mask >= 0.5).float()
            # from IPython import embed;embed()
            patch_imgs, patch_masks, patch_fine_probs, patch_coors = \
            self.get_local_input(img, ori_size_mask, fine_probs, patch_size)
            if patch_imgs is None:
                return ori_size_mask
            
            batch_max = 1
            num_ins = len(patch_imgs)
            if num_ins <= batch_max:
                xs = [(patch_masks, patch_imgs, patch_fine_probs)]
            else:
                xs = []
                for idx in range(0, num_ins, batch_max):
                    end = min(num_ins, idx + batch_max)
                    xs.append((patch_masks[idx: end], patch_imgs[idx:end], patch_fine_probs[idx:end]))
            local_masks, _ = self.p_sample_loop(xs, 
                                                local_indices, 
                                                patch_imgs.device,
                                                use_last_step=True)
            
            mask = self.paste_local_patch(local_masks, ori_size_mask, patch_coors)
            return mask

        else:
            return F.interpolate(model_size_mask, size=ori_shape)

    def get_local_input(self, img, ori_size_mask, fine_probs, model_size):
        ori_shape = img.shape[-3:]
        img_h, img_w, img_d = ori_shape
        ori_size_fine_probs = F.interpolate(fine_probs, ori_shape)
        fine_prob_thr = 0.9
        fine_prob_thr = fine_probs.max().item() * fine_prob_thr
        low_cofidence_points = fine_probs < fine_prob_thr
        scores = fine_probs[low_cofidence_points]
        y_c, x_c, z_c = torch.where(low_cofidence_points.squeeze(0).squeeze(0))
        scale_factor_y, scale_factor_x, scale_factor_d = img_h / model_size[0], img_w / model_size[1], img_d / model_size[2]
        y_c, x_c, z_c = (y_c * scale_factor_y).int(), (x_c * scale_factor_x).int(), (z_c * scale_factor_d).int()            
        scores = 1 - scores # 越高代表预测越不准确
        # in orig size
        patch_coors = _get_patch_coors(x_c, y_c, z_c, 0, 0, 0, img_w, img_h, img_d, model_size, scores, 0.3)
        return self.crop_patch(img, ori_size_mask, ori_size_fine_probs, patch_coors)
    
    def get_global_input(self, img, coarse_masks, model_size, current_device):
        global_img = F.interpolate(img, size=model_size, mode='trilinear')
        global_mask = F.interpolate(coarse_masks, size=model_size, mode='nearest').to(current_device)
        global_mask = (global_mask >= 0.5).float()
        return global_img, global_mask    
    
    def crop_patch(self, img, mask, fine_probs, patch_coors):
        patch_imgs, patch_masks, patch_fine_probs, new_patch_coors = [], [], [], []
        for coor in patch_coors:
            patch_mask = mask[:, :, coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]]
            # if (patch_mask.any()) and (not patch_mask.all()): #全是0的patch是怎么来的，按道理不会被采集到
            patch_imgs.append(img[:, :, coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]])
            patch_fine_probs.append(fine_probs[:, :, coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]])
            patch_masks.append(patch_mask)
            new_patch_coors.append(coor)
        if len(patch_imgs) == 0:
            return None, None, None, None
        patch_imgs = torch.cat(patch_imgs, dim=0)
        patch_masks = torch.cat(patch_masks, dim=0)
        patch_fine_probs = torch.cat(patch_fine_probs, dim=0)
        patch_masks = (patch_masks >= 0.5).float()
        return patch_imgs, patch_masks, patch_fine_probs, new_patch_coors
    
    def paste_local_patch(self, local_masks, mask, patch_coors, label = None):
        mask = mask.squeeze(0).squeeze(0)
        refined_mask = torch.zeros_like(mask)
        weight = torch.zeros_like(mask)
        local_masks = local_masks.squeeze(1)
        pred_res = 0.
        coarse_res = 0.
        for local_mask, coor in zip(local_masks, patch_coors):
            if label is not None:
                local_label = label[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]]
                local_coarse = mask[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]]
                pred_res_ = self.cal_dice(local_label,local_mask)
                coarse_res_ = self.cal_dice(local_label,local_coarse)
                print(coarse_res_,'-->',pred_res_)
                pred_res += pred_res_
                coarse_res += coarse_res_
                
            refined_mask[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]] += local_mask
            weight[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]] += 1
        if label is not None:
            print("TOTAL:",coarse_res/len(patch_coors),pred_res/len(patch_coors))
        refined_area = (weight > 0).float()
        weight[weight == 0] = 1
        refined_mask = refined_mask / weight
        refined_mask = (refined_mask >= 0.5).float()
        return refined_area * refined_mask + (1 - refined_area) * mask
    
def _get_patch_coors(x_c, y_c, z_c, X_1, Y_1, Z_1, X_2, Y_2, Z_2, patch_size, scores, thresh):
        # model size patch
        z_1, z_2 = z_c - patch_size[2]/2, z_c + patch_size[2]/2
        y_1, y_2 = y_c - patch_size[0]/2, y_c + patch_size[0]/2
        x_1, x_2 = x_c - patch_size[1]/2, x_c + patch_size[1]/2
        
        invalid_z = z_1 < Z_1
        z_1[invalid_z] = Z_1
        z_2[invalid_z] = Z_1 + patch_size[2]
        invalid_z = z_2 > Z_2
        z_1[invalid_z] = Z_2 - patch_size[2]
        z_2[invalid_z] = Z_2
        
        invalid_y = y_1 < Y_1
        y_1[invalid_y] = Y_1
        y_2[invalid_y] = Y_1 + patch_size[0]
        invalid_y = y_2 > Y_2
        y_1[invalid_y] = Y_2 - patch_size[0]
        y_2[invalid_y] = Y_2
        
        invalid_x = x_1 < X_1
        x_1[invalid_x] = X_1
        x_2[invalid_x] = X_1 + patch_size[1]
        invalid_x = x_2 > X_2
        x_1[invalid_x] = X_2 - patch_size[1]
        x_2[invalid_x] = X_2
        
        proposals = torch.stack((x_1, y_1, z_1, x_2, y_2, z_2), dim=1)
        patch_coors = aligned_3d_nms(proposals, 
                                     scores, 
                                     classes = torch.ones(len(proposals),device=proposals.device),
                                     thresh=thresh)
        return patch_coors.int()