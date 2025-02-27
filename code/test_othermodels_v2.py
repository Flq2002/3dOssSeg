"""
bash train_othermodels.sh -c 4 -e swin_unetr -t oss1 -p 64 -d flase -f /media/HDD/fanlinqian/work_dirs_othermodels/Exp_OSS/20241103-1730641011-unetr
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
parser.add_argument('--full_path', type=str)
parser.add_argument('--timestep', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_name', type=str,default='unert')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
patch_size = (args.patch_size,args.patch_size,args.patch_size)
import torch
from utils import read_list, maybe_mkdir, fetch_data
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
            is_coarse = True,
            data_type='data2' if args.task == 'oss2' else 'data1',
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
            oss_tag = "DG",
            data_type='data2' if args.task == 'oss2' else 'data1',
            model="other",
            is_coarse='false',
            base_model = "new_manual1"
        )
        return DataLoader(dst,num_workers=2)


from monai import transforms
def transform_val_oss(patch_size = (32,32,32)):
    am = True
    val_transform = transforms.Compose(
            [   
                transforms.CropForegroundd(keys=["image", "label","coarse_label","ref_label"], source_key="ref_label", allow_missing_keys = am, margin=32 if args.patch_size==64 else 8),
                transforms.CenterSpatialCropd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, roi_size=patch_size),
                transforms.SpatialPadd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, spatial_size=patch_size),
                # transforms.NormalizeIntensityd(keys="image", nonzero=True, allow_missing_keys = am, channel_wise=True),
                transforms.ToTensord(keys=["image", "label", "coarse_label","ref_label"],allow_missing_keys = am, dtype=torch.float),
            ]
        )
    return val_transform

import shutil
def savenii(data,type,data_id):
    out = sitk.GetImageFromArray(data)
    sitk.WriteImage(out, f'{test_save_path}/{data_id}_{type}.nii.gz')
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
post_label = AsDiscrete(to_onehot=True, n_classes=config.num_cls)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=config.num_cls)
dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
def cal_dice(logits,target):
    val_labels_list = decollate_batch(target)
    val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
    val_outputs_list = decollate_batch(logits)
    val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
    acc = dice_acc(y_pred=val_output_convert, y=val_labels_convert)
    acc_list = acc.detach().cpu().numpy()
    avg_acc = np.mean([np.nanmean(l) for l in acc_list])
    return avg_acc
from math import log10, sqrt
def snr(image):
    # signal_power = np.mean(image ** 2)
    # noise_power = np.var(image)
    # return 10 * log10(signal_power / noise_power)
    return np.mean(image)/np.std(image)
def contrast(image):
    return np.std(image)
def uniformity(image):
    return np.var(image) / np.mean(image)

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
    snapshot_path =  args.full_path
    test_save_path =  os.path.join(snapshot_path + '/predictions/')
    maybe_mkdir(test_save_path)
    if args.model_name == 'unetr':
        from Othermodels.Unetr import UNETR
        model = UNETR(
                in_channels=1,
                out_channels=config.num_cls,
                img_size=patch_size,
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed='perceptron',
                norm_name='instance',
                conv_block=True,
                res_block=True,
                dropout_rate=0.0,
            ).cuda()
    elif args.model_name == "swin_unetr":
        from Othermodels.swinUnetr import SwinUNETR
        model = SwinUNETR(
                img_size=patch_size,
                in_channels=1,
                out_channels=config.num_cls,
                feature_size=12,
                use_checkpoint=True,
            ).cuda()
    elif args.model_name == "Vnet":
        from Othermodels.Vnet import VNet
        model = VNet().cuda()
    else:
        raise ImportError
    shutil.copy('/home/suk/3dOssSeg/code/test_othermodels_v2.py', snapshot_path+'/test.py')
    eval_loader = make_loader(args.split, task=args.task, is_training=False, transforms_tr = transform_val_oss(patch_size))
    
    for it in np.arange(600,5400,600):
        ckpt_path = os.path.join(snapshot_path, f'ckpts/model_iter{it}.pth')
    # for i in range(1):
    #     ckpt_path = os.path.join(snapshot_path, f'ckpts/model_iter5400.pth')
        eval_list = read_list(args.split,args.task)
        idx=-1
        print(f'load checkpoint from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        model.eval()
        dice_list=[]
        cal_snr=0
        cal_con=0
        cal_uni=0

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                idx+=1
                image, label = fetch_data(batch)
                label = label.long()
                # continue
                
                preds = model(image)
                dice = cal_dice(preds, label)
                # # print(dice)
                dice_list.append(dice.item())
                savenii((preds.argmax(1)).squeeze().cpu().numpy().astype(np.int8),type = "pred",data_id=eval_list[idx])
                
                savenii(label.squeeze().cpu().numpy().astype(np.int8),type = "label",data_id=eval_list[idx])
                savenii(image.squeeze().cpu().numpy(),type = "image",data_id=eval_list[idx])
                # tmpp = snr(image.squeeze().cpu().numpy())
                # print(eval_list[idx],":",tmpp)
                # cal_snr+=tmpp
                # break
        # print(cal_snr/len(eval_loader))
        print("TOTAL:",np.mean(dice_list))
