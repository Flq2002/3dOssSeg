"""
bash train_refiner.sh -c 2 -t la -d flase -s 6 -b /media/HDD/fanlinqian/work_dirs_othermodels/Exp_LA/20240914-1726327487-unert -r /media/HDD/fanlinqian/work_dirs_ssl/Exp_refiner_LA/20240917-1726543572-diffusion_refiner_basemodel/
"""

import os,sys
sys.path.append('/home/suk/3dOssSeg/code/utils')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='0')
parser.add_argument('--refiner_path', type=str)
parser.add_argument('--basemodel_path', type=str)
parser.add_argument('--timestep', type=int, default=6)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from utils import read_list, maybe_mkdir
from utils import config
import numpy as np
from tqdm import tqdm
from utils import read_data_coarse
import SimpleITK as sitk
from Segrefiner.segrefiner_base import SegRefiner
config = config.Config(args.task)


# def test_all_case(basemodel_path, task, net, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
#     for data_id in tqdm(ids_list):
#         image, coarse_label = read_data_coarse(data_id, task=task, full_path=basemodel_path, normalize=True)

#         pred, _ = test_single_case(
#             net,
#             coarse_label,
#             image,
#             stride_xy,
#             stride_z,
#             patch_size,
#             num_classes=num_classes
#         )
#         out = sitk.GetImageFromArray(pred.astype(np.float32))
#         sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

# import math
# def test_single_case(net:SegRefiner, coarse_label, image, stride_xy, stride_z, patch_size, num_classes):

#     padding_flag = image.shape[0] < patch_size[0] or image.shape[1] < patch_size[1] or image.shape[2] < patch_size[2]
#     if padding_flag:
#         pw = max((patch_size[0] - image.shape[0]) // 2 + 1, 0)
#         ph = max((patch_size[1] - image.shape[1]) // 2 + 1, 0)
#         pd = max((patch_size[2] - image.shape[2]) // 2 + 1, 0)
#         image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#         coarse_label = np.pad(coarse_label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

#     image = image[np.newaxis]
#     coarse_label = coarse_label[np.newaxis]
#     _, dd, ww, hh = image.shape


#     image = image.transpose(0, 3, 2, 1) # <-- take care the shape
#     coarse_label = coarse_label.transpose(0, 3, 2, 1)
#     patch_size = (patch_size[2], patch_size[1], patch_size[0])
#     _, ww, hh, dd = image.shape

#     sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
#     sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
#     sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

#     score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
#     cnt = np.zeros(image.shape[1:4]).astype(np.float32)
#     # print("score_map", score_map.shape)
#     for x in range(sx):
#         xs = min(stride_xy*x, ww-patch_size[0])
#         for y in range(sy):
#             ys = min(stride_xy*y, hh-patch_size[1])
#             for z in range(sz):
#                 zs = min(stride_z*z, dd-patch_size[2])
#                 test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
#                 coarse_patch = coarse_label[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
#                 # print("test", test_patch.shape)
#                 test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
#                 test_patch = torch.from_numpy(test_patch).cuda()
#                 coarse_patch = np.expand_dims(coarse_patch, axis=0)
#                 coarse_patch = torch.from_numpy(coarse_patch).cuda().long()

#                 # print("===",test_patch.size())
#                 # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
#                 test_patch = test_patch.transpose(2, 4)
#                 coarse_patch = coarse_patch.transpose(2, 4)
#                 # y1, _, _, _ = net(test_patch) # <--
#                 y1 = net.single_test(test_patch,coarse_patch)
#                 y = torch.cat((y1,1.-y1),dim=1)
#                 y = y.cpu().data.numpy()
#                 y = y[0, ...]
#                 y = y.transpose(0, 3, 2, 1)
#                 score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
#                 cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

#     score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
#     score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
#     label_map = np.argmax(score_map, axis=0)
#     return label_map, score_map


def test_all_case(basemodel_path, task, net:SegRefiner, ids_list, test_save_path=None):
    print(f'load coarse label from {basemodel_path}')
    for data_id in tqdm(ids_list):
        image, coarse_label,_ = read_data_coarse(data_id, task=task, full_path=basemodel_path, norm_cfg=config.norm_cfg)
        # from IPython import embed;embed()
        pred = net.single_test(image[None,None,...], 
                               coarse_label[None,None,...], 
                               config.patch_size,
                            #    need_local_step=True
                               ).squeeze().cpu().data.numpy()
        out = sitk.GetImageFromArray((pred>=0.5).astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')


diffusion_cfg=dict(
    betas=dict(
        type='linear',
        start=0.8,
        stop=0,
        num_timesteps=args.timestep),
    diff_iter=False)
denoise_cfg=dict(
    in_channels=2,
    out_channels=1,
    # model_channels=128,
    model_channels=64,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    attention_strides=(16, 32),
    learn_time_embd=True,
    channel_mult = (1, 1, 2, 2, 4, 4),
    # dropout=0.0,
    dropout=0.1,
    dims = 3,
    num_timesteps=args.timestep)
import shutil
if __name__ == '__main__':
    stride_dict = {
        0: (16, 4),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]
    snapshot_path =  args.refiner_path
    test_save_path =  os.path.join(snapshot_path + '/predictions/')
    maybe_mkdir(test_save_path)
    model = SegRefiner(
        diffusion_cfg=diffusion_cfg,
        denoise_cfg=denoise_cfg
    ).cuda()
    shutil.copy('/home/suk/3dOssSeg/code/test_diffusion_refiner.py', snapshot_path+'/test.py')

    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
    print(f'load checkpoint from {ckpt_path}')
    with torch.no_grad():
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        model.eval()
        test_all_case(
            args.basemodel_path,
            args.task,
            model,
            read_list(args.split, task=args.task),
            test_save_path=test_save_path
        )
