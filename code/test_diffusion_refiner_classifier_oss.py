"""
bash train_refiner.sh -c 5 -s 6 -p 64 -e diffusion_refiner_classifier_oss -t oss1 -l 5e-5 -n 2000 -d flase -r /media/HDD/fanlinqian/work_dirs_refiner/Exp_refiner_OSS/20241108-1731037232-diffusion_refiner_oss -b /media/HDD/fanlinqian/work_dirs_othermodels/Exp_OSS/20241105-1730777670-swin_unetr
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
parser.add_argument('--timestep', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--base_model', type=str, default=None)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
patch_size = (args.patch_size,args.patch_size,args.patch_size)
import torch
from utils import read_list, maybe_mkdir, fetch_data_oss
from utils import config
import numpy as np
from tqdm import tqdm
from utils import read_data_coarse,seed_worker
import SimpleITK as sitk
# from Segrefiner.segrefiner_base import SegRefiner as SegRefiner_oss
from Segrefiner.segrefiner_oss import SegRefiner_oss
config = config.Config(args.task)


from data.data_loaders import DatasetOss
from torch.utils.data import DataLoader
def make_loader(split, dst_cls=DatasetOss, repeat=None, is_training=True, unlabeled=False, task="", transforms_tr=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls,
            # is_coarse = True,
            data_type='data2' if args.task == 'oss2' else 'data1'
        )
        return DataLoader(
            dst,
            batch_size=4,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms_tr,
            is_coarse='false',
            # is_coarse="random_0",
            oss_tag = "DG",
            data_type='data2' if args.task == 'oss2' else 'data1',
            # data_type='data4',
            base_model=args.base_model,
        )
        return DataLoader(dst,num_workers=2)


def test_all_case(basemodel_path, task, net:SegRefiner_oss, ids_list, test_save_path=None):
    print(f'load coarse label from {basemodel_path}')
    for data_id in tqdm(ids_list):
        image, coarse_label,_ = read_data_coarse(data_id, task=task, full_path=basemodel_path, norm_cfg=config.norm_cfg)
        # from IPython import embed;embed()
        pred = net.single_test(image[None,None,...], 
                               coarse_label[None,None,...], 
                                patch_size,
                            #    need_local_step=True
                               ).squeeze().cpu().data.numpy()
        out = sitk.GetImageFromArray((pred>=0.5).astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

from monai import transforms
def transform_val_oss(patch_size = (32,32,32)):
    
    val_transform = transforms.Compose(
        [   
            transforms.CropForegroundd(keys=["image", "label","coarse_label", "ref_label"], source_key="ref_label", margin=32 if args.patch_size==64 else 8, allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=["image", "label","coarse_label","guided_image"], roi_size=patch_size, allow_missing_keys=True),
            transforms.SpatialPadd(keys=["image", "label","coarse_label","guided_image"], spatial_size=patch_size, allow_missing_keys=True),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True, allow_missing_keys=True),
            transforms.ToTensord(keys=["image", "label", "coarse_label","guided_image"],dtype=torch.float, allow_missing_keys=True),
        ]
    )
    return val_transform

def transform_val_oss_other(patch_size = (32,32,32)):
    val_transform = transforms.Compose(
        [   
            transforms.CropForegroundd(keys=["image", "label", "ref_label"], source_key="ref_label", margin=32 if args.patch_size==64 else 8),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "coarse_label"],dtype=torch.float),
        ]
    )
    return val_transform
from data.other_process.postprocess_dia_moban import dilate_slice
from scipy.ndimage import binary_closing
def mor_dilate_slice(slice):
    dilated_slice = dilate_slice(slice)
    dilated_slice = binary_closing(dilated_slice, iterations=2)
    dilated_slice = dilate_slice(dilated_slice)
    return dilated_slice

def dilate_volume(volume):
    for j in range(volume.shape[1]):
        volume[:,j,:] = mor_dilate_slice(volume[:,j,:])
    return volume

diffusion_cfg=dict(
    betas=dict(
        type='linear',
        start=0.8,
        stop=0,
        num_timesteps=args.timestep),
    diff_iter=False)
denoise_cfg=dict(
    # in_channels=2+3,
    in_channels=1,
    out_channels=1,
    model_channels=128,
    # model_channels=64,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    attention_strides=(16, 32),
    learn_time_embd=True,
    channel_mult = (1,1,2,2,4,4),
    dropout=0.0,
    # dropout=0.1,
    dims = 3,
    num_timesteps=args.timestep,
    use_img_embd = 'layers_orig',
    )
import shutil
def savenii(data,type,data_id,sta=None,end=None,orig_size = None):
    if sta is not None:
        center = (sta+end)//2
        # from IPython import embed;embed()
        sta = center - 32
        end = center + 32
        data_pad = np.zeros(orig_size,dtype=data.dtype)
        try:
            data_pad[sta[0]:end[0],sta[1]:end[1],sta[2]:end[2]] = data
        except Exception as e:
            print(e)
            print(data_id)
            return
        data = data_pad
    out = sitk.GetImageFromArray(data)
    sitk.WriteImage(out, f'{test_save_path}/{data_id}_{type}.nii.gz')
if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    stride_dict = {
        0: (16, 4),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]
    snapshot_path =  args.refiner_path
    test_save_path =  os.path.join(snapshot_path + '/predictions/')
    maybe_mkdir(test_save_path)
    model = SegRefiner_oss(
        diffusion_cfg=diffusion_cfg,
        denoise_cfg=denoise_cfg
    ).cuda()
    
    from Othermodels.resnet18 import ResNet3D_18
    classifier = ResNet3D_18(num_classes=1).cuda()
    
    shutil.copy('/home/suk/3dOssSeg/code/test_diffusion_refiner_classifier_oss.py', snapshot_path+'/test.py')
    if args.base_model is None or "manual" in args.base_model:
        eval_loader = make_loader(args.split, task=args.task, is_training=False, transforms_tr = transform_val_oss(patch_size))
    else:
        eval_loader = make_loader(args.split, task=args.task, is_training=False, transforms_tr = transform_val_oss_other(patch_size))
    ckpt_path = os.path.join(snapshot_path, f'ckpts/model_iter52000.pth')
    eval_list = read_list(args.split,args.task)
    idx=-1
    print(f'load checkpoint from {ckpt_path}')
    # for epo in np.arange(10400,100000,10400):
    for epo in ["final_model"]:
        ckpt_path = os.path.join(snapshot_path, f'ckpts/{epo}.pth')
        with torch.no_grad():
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
            classifier.load_state_dict(torch.load(f"/media/HDD/fanlinqian/work_dirs_classifier/Exp_refiner_OSS/20250126-1737897626-classifier/ckpts/model_95.pth")["state_dict"])
            model.eval()
            classifier.eval()
            test_dice, test_iou = 0.,0.
            cl_dice, cl_iou = 0.,0.
            tot_prob = 0.
            with torch.no_grad():
                for batch in tqdm(eval_loader):
                    idx+=1
                    image, label, coarse_label = fetch_data_oss(batch)
                    # from IPython import embed;embed()
                    # guided_image = batch["guided_image"]
                    # savenii(guided_image,"guided_image",eval_list[idx])
                    # continue
                    
                    pred = model.single_test(image,coarse_label,patch_size,save_diff=False)
                    
                    prob = classifier((pred>=0.5).float()).squeeze().sigmoid()
                    # print(prob)
                    tot_prob += prob
                    # pred,diffu_x = model.single_test(image,coarse_label,patch_size,save_diff=True)
                    # for t,xx in enumerate(diffu_x):
                    #     # if t in [7,8]:
                    #     # savenii((xx>=0.5).astype(np.int8),type = f"diffu{t}",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0])
                    #     savenii((xx>=0.5).astype(np.int8),type = f"diffu{t}",data_id=eval_list[idx])
                    # from IPython import embed;embed()
                    
                    # dice,iou=model.cal_dice(label,pred).item(),model.cal_iou(label,pred).item()
                    # dice_cl,iou_cl=model.cal_dice(label,coarse_label).item(),model.cal_iou(label,coarse_label).item()
                    # cl_dice += dice_cl
                    # cl_iou  += iou_cl
                    # test_dice += dice
                    # test_iou += iou
                    # # print(iou_cl,'-->',iou)
                    # print(dice_cl,'-->',dice)

                    
                    
                    # savenii((pred>=0.5).squeeze().cpu().numpy().astype(np.int8),type = "pred",data_id=eval_list[idx])
                    # savenii(label.squeeze().cpu().numpy(),type = "label",data_id=eval_list[idx])
                    # savenii(image.squeeze().cpu().numpy(),type = "image",data_id=eval_list[idx])
                    # savenii(coarse_label.squeeze().cpu().numpy(),type = "coarse",data_id=eval_list[idx])
                    # from IPython import embed;embed()
                    # for t,xx in enumerate(batch["seg_mid"].squeeze(0)):
                    #     print(xx.sum())
                    #     savenii(xx.numpy().astype(np.int8),type = f"mid{t}",data_id=eval_list[idx])

                    # dilate = (pred>=0.5).squeeze().cpu().numpy().astype(np.int8)
                    # dilate = dilate_volume(dilate)
                    # from IPython import embed;embed()
                    # savenii(dilate,type = "dilate",data_id=eval_list[idx])
                    # savenii(dilate,type = "dilate",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                    
                    # savenii((pred>=0.5).squeeze().cpu().numpy().astype(np.int8),type = "pred",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                    # savenii(label.squeeze().cpu().numpy(),type = "label",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0])
                    # savenii(coarse_label.squeeze().cpu().numpy(),type = "orig_label",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                    
                    # break
            #     test_dice /= len(eval_loader)
            #     test_iou /= len(eval_loader)
            #     cl_dice /= len(eval_loader)
            #     cl_iou /= len(eval_loader)
            # print("dice:",cl_dice, "-->",test_dice)
            # print("coarse:",cl_dice,cl_iou)
            print(epo,":average",tot_prob/len(eval_loader))