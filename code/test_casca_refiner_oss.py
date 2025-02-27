"""
bash train_refiner.sh -c 5 -s 6 -p 64 -e casca_refiner_oss -t oss1 -l 5e-5 -n 2000 -d flase -r /media/HDD/fanlinqian/work_dirs_casca/Exp_refiner_OSS/20250221-1740129839-casca -b new_manual1
bash train_refiner_oss2.sh -c 5 -s 6 -p 64 -e diffusion_refiner_oss -t oss2 -l 5e-5 -n 2000 -d flase -r /media/HDD/fanlinqian/work_dirs_refiner/Exp_refiner_OSS/20250205-1738766352-diffusion_refiner_oss -b /media/HDD/fanlinqian/work_dirs_othermodels/Exp_OSS/20250218-1739846623-swin_unetr
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
from cascadpsp.models.psp.pspnet import PSPNet
from cascadpsp.models.sobel_op import SobelComputer
from cascadpsp.util.metrics_compute import compute_loss_and_metrics,iou_hooks_to_be_used
from cascadpsp.util.logger import BoardLogger
from cascadpsp.util.log_integrator import Integrator
from cascadpsp.eval_helper import safe_forward
config = config.Config(args.task)


from data.data_loaders import DatasetOss,DatasetOss_noise
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

def cal_dice(target, mask, eps=1e-3):
    target = target.clone().detach() >= 0.5
    mask = mask.clone().detach() >= 0.5
    si = (target & mask).sum(-1).sum(-1).sum(-1)
    return (2 * si) / (target.sum(-1).sum(-1).sum(-1) + mask.sum(-1).sum(-1).sum(-1) + eps)
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
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50').cuda()
    shutil.copy('/home/suk/3dOssSeg/code/test_casca_refiner_oss.py', snapshot_path+'/test.py')
    if args.base_model is None or "manual" in args.base_model:
        eval_loader = make_loader(args.split, task=args.task, is_training=False, transforms_tr = transform_val_oss(patch_size))
    else:
        eval_loader = make_loader(args.split, task=args.task, is_training=False, transforms_tr = transform_val_oss_other(patch_size))
    # for it in np.arange(10200,91800,10200):
    #     ckpt_path = os.path.join(snapshot_path, f'ckpts/model_iter{it}.pth')
    for it in [61200]:
        ckpt_path = os.path.join(snapshot_path, f'ckpts/model_iter12800.pth')
        eval_list = read_list(args.split,args.task)
        idx=-1
        print(f'load checkpoint from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        model.eval()
        test_dice, test_iou = 0.,0.
        cl_dice, cl_iou = 0.,0.
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                idx+=1
                image, label, coarse_label = fetch_data_oss(batch)
                # from IPython import embed;embed()
                # guided_image = batch["guided_image"]
                # savenii(guided_image,"guided_image",eval_list[idx])
                # continue
                images = safe_forward(model, image, coarse_label)
                pred = images["pred_224"]
                # pred,diffu_x = model.single_test(image,coarse_label,patch_size,save_diff=True)
                # for t,xx in enumerate(diffu_x):
                #     # if t in [7,8]:
                #     # savenii((xx>=0.5).astype(np.int8),type = f"diffu{t}",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0])
                #     savenii((xx>=0.5).astype(np.int8),type = f"diffu{t}",data_id=eval_list[idx])
                # from IPython import embed;embed()
                
                dice=cal_dice(label,pred).item()
                dice_cl=cal_dice(label,coarse_label).item()
                cl_dice += dice_cl
                test_dice += dice
                
                # # # print(iou_cl,'-->',iou)
                # print(dice_cl,'-->',dice)

                savenii((pred>=0.5).squeeze().cpu().numpy().astype(np.int8),type = "pred",data_id=eval_list[idx])
                savenii(label.squeeze().cpu().numpy(),type = "label",data_id=eval_list[idx])
                savenii(image.squeeze().cpu().numpy(),type = "image",data_id=eval_list[idx])
                savenii(coarse_label.squeeze().cpu().numpy(),type = "coarse",data_id=eval_list[idx])
                # from IPython import embed;embed()
                # for t,xx in enumerate(batch["seg_mid"].squeeze(0)):
                #     print(xx.sum())
                #     savenii(xx.numpy().astype(np.int8),type = f"mid{t}",data_id=eval_list[idx])

                # dilate = (pred>=0.5).squeeze().cpu().numpy().astype(np.int8)
                # dilate = dilate_volume(dilate)
                # filter_pred = keep_largest_components(dilate,1)
                # savenii(dilate,type = "dilate",data_id=eval_list[idx])
                # savenii(filter_pred,type = "dilate_filter",data_id=eval_list[idx])
                # savenii(dilate,type = "dilate",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                # savenii(filter_pred,type = "B",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                # savenii((pred>=0.5).squeeze().cpu().numpy().astype(np.int8),type = "pred",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                # savenii(label.squeeze().cpu().numpy(),type = "label",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0])
                # savenii(coarse_label.squeeze().cpu().numpy(),type = "orig_label",data_id=eval_list[idx],sta=batch["foreground_start_coord"][0],end=batch["foreground_end_coord"][0],orig_size = batch["image_transforms"][0]["orig_size"])
                
                # break
            test_dice /= len(eval_loader)
            test_iou /= len(eval_loader)
            cl_dice /= len(eval_loader)
            cl_iou /= len(eval_loader)
        print("dice:",cl_dice, "-->",test_dice)