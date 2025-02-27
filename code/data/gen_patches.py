# -----------------------------------------------------------------------------------
# References:
# https://github.com/chenhang98/BPR/blob/main/tools/split_patches.py#L83
# -----------------------------------------------------------------------------------


import os,sys
sys.path.append('/home/suk/GenericSSL/code/')
from tqdm import tqdm
from utils import read_list,read_nifti
from utils.config import Config
import numpy as np
import torch
import torch.nn.functional as F
from Segrefiner.segrefiner_base import _get_patch_coors
import SimpleITK as sitk

def find_float_boundary(maskdt, width=3):
    # Extract boundary from instance mask
    if len(maskdt.shape) == 3:
        maskdt = torch.Tensor(maskdt).unsqueeze(0).unsqueeze(0)
    boundary_finder = maskdt.new_ones((1, 1, width, width, width))
    boundary_mask = F.conv3d(maskdt, boundary_finder,
                             stride=1, padding=width//2)
    bml = torch.abs(boundary_mask - width**3)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width**3/2)
    return fbmask[0, 0]

def get_dets(mask,crop_size,iou_thres=0.2):
    fbmask = find_float_boundary(mask)
    y,x,z = torch.where(fbmask)
    scores = fbmask[y,x,z]
    if len(mask.shape) == 5:
        mask = mask.squeeze(0).squeeze(0)
    dets = _get_patch_coors(
        x,y,z,
        0,0,0,
        mask.shape[1],mask.shape[0],mask.shape[2],
        crop_size,scores,iou_thres
    )
    return dets

def crop_patch(img, lb, cl, patch_coors):
    img_pc, lb_pc, cl_pc = [],[],[]
    for coor in patch_coors:
        img_pc.append(img[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]])
        lb_pc.append(lb[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]])
        cl_pc.append(cl[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]])
    return img_pc, lb_pc, cl_pc
def save_itk(data,path):
    out = sitk.GetImageFromArray(data.astype(np.float32))
    sitk.WriteImage(out, path)

if __name__ == '__main__':
    task  = 'la'
    config = Config(task)
    full_path = '/media/HDD/fanlinqian/work_dirs_othermodels/Exp_LA/20240914-1726327487-unert/'
    cropH,cropW,cropD = 32,32,16
    # cropH,cropW,cropD = 64,64,32
    
    ids_list = read_list('train_0.7',task = task)
    patch_ids_list = []
    for data_id in tqdm(ids_list):
        im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')
        cl_path = os.path.join(full_path, 'predictions', f'{data_id}.nii.gz')
        lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')

        image = np.load(im_path)
        h,w,d = image.shape
        if h < cropH or w < cropW or d < cropD:
            continue

        label = np.load(lb_path)
        coarse_label = read_nifti(cl_path)
        dets = get_dets(coarse_label,(cropW,cropH,cropD)).numpy()
        # for coor in dets:
        #     coarse_label[coor[1]:coor[4], coor[0]:coor[3], coor[2]:coor[5]]=2
        # print(data_id)
        # print(len(dets))
        # save_itk(coarse_label,os.path.join(config.save_dir, 'npy_patch', f'{data_id}_tmp.nii.gz'))
        # exit(0)

        img_pc, lb_pc, cl_pc = crop_patch(image,label,coarse_label,dets)

        for idx,(img,lb,cl) in enumerate(zip(img_pc, lb_pc, cl_pc)):
            patch_ids_list.append(f'{data_id}_{idx}_')
            np.save(
                os.path.join(config.save_dir, f'npy_patch{cropH}_0.7', f'{data_id}_{idx}_img.npy'),
                img
            )
            np.save(
                os.path.join(config.save_dir, f'npy_patch{cropH}_0.7', f'{data_id}_{idx}_lb.npy'),
                lb
            )
            np.save(
                os.path.join(config.save_dir, f'npy_patch{cropH}_0.7', f'{data_id}_{idx}_cl.npy'),
                cl
            )
            # save_itk(
            #     img,
            #     os.path.join(config.save_dir, 'npy_patch', f'{data_id}_{idx}_img.nii.gz'),
            # )
            # save_itk(
            #     lb,
            #     os.path.join(config.save_dir, 'npy_patch', f'{data_id}_{idx}_lb.nii.gz'),
            # )
            # save_itk(
            #     cl,
            #     os.path.join(config.save_dir, 'npy_patch', f'{data_id}_{idx}_cl.nii.gz'),
            # )
            
    with open(config.save_dir+f'/split_txts/patch{cropH}_train_0.7.txt','w') as f:
        for l in patch_ids_list:
            f.write(l+'\n')
    
        


        
        





