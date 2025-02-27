"""
bash train_othermodels.sh -c 4 -e swin_unetr -t oss1 -l 1e-2 -n 1000 -d true -p 64
"""

import os
import sys
import logging
from tqdm import tqdm
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='synapse', required=True)
parser.add_argument('--exp', type=str, default='diffusion')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, required=True)
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=False)
parser.add_argument('-ep', '--max_epoch', type=int, default=300)   
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--model_name', type=str)
parser.add_argument('--need_val', action='store_true', default=True)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
patch_size = (args.patch_size,args.patch_size,args.patch_size)
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from Segrefiner.segrefiner_oss import SegRefiner_oss
# from Segrefiner.segrefiner_base import SegRefiner as Seg   efiner_oss
from utils import EMA, maybe_mkdir, get_lr, fetch_data, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetOss,DatasetOss_noise
from utils.config import Config
from utils import read_list,read_data_coarse
from Othermodels.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.lr_scheduler import ConstantWarmupCosineAnnealingLR
from monai.losses import DiceCELoss, DiceLoss
config = Config(args.task)

def make_loader(split, dst_cls=DatasetOss, repeat=None, is_training=True, unlabeled=False, task="", transforms_tr=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls,
            # is_coarse = False,
            is_coarse="false",
            data_type='data2' if args.task == 'oss2' else 'data1',
            model="other",
            base_model = "new_manual1",
            labelonehot = True if args.model_name == "diffunet" else False
        )
        return DataLoader(
            dst,
            batch_size=8,
            shuffle=True,
            num_workers=args.num_workers,
            # num_workers = 1,
            pin_memory=True,
            drop_last=True
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms_tr,
            # is_coarse = 'noise',
            is_coarse="false",
            data_type='data2' if args.task == 'oss2' else 'data1',
            model="other",
            base_model = "new_manual1",
            labelonehot = True if args.model_name == "diffunet" else False
        )
        return DataLoader(dst, pin_memory=True,num_workers=2,batch_size=8)

def make_model_all():
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
    elif args.model_name == "swin_unetrv2":
        from Othermodels.swinUnetrv2 import SwinUNETR
        model = SwinUNETR(
            img_size=patch_size,
            in_channels=1,
            out_channels=config.num_cls,
            feature_size=12,
            use_checkpoint=True,
            use_v2=True
        ).cuda()
    elif args.model_name == "MedNext":
        from Othermodels.mednextv1.MedNextV1 import MedNeXt
        model = MedNeXt(
          in_channels=1,
          n_channels=2,
          n_classes=config.num_cls, 
          deep_supervision=False,
          do_res=True,
          do_res_up_down=True,
          dim = '3d',
        ).cuda()
    elif args.model_name == "diffunet":
        from Othermodels.diff_unet.DiffUnet import DiffUNet
        model = DiffUNet().cuda()
    elif args.model_name == "nnunet":
        from Othermodels.nnUNet.nnunet.network_architecture.generic_UNet import Generic_UNet
        from Othermodels.nnUNet.nnunet.network_architecture.initialization import InitWeights_He
        from torch import nn
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # infer
        base_num_features = 30
        len_net_num_pool_op_kernel_sizes = 3
        conv_per_stage = 2
        net_conv_kernel_sizes = []
        model = Generic_UNet(1, base_num_features, 2,
                             len_net_num_pool_op_kernel_sizes,
                             conv_per_stage, 2, nn.Conv3d, nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d,
                             dropout_op_kwargs,
                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                             len_net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True
                             ).cuda()
    else:
        raise ImportError
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=1e-5,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    return model, optimizer



from monai import transforms
# def transform_train_oss(patch_size = (32,32,32)):
#     am = True
#     tr_transforms =transforms.Compose([ #TODO add other aug
#             transforms.CropForegroundd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, source_key="ref_label", margin=16 if args.patch_size==64 else 8),

#             transforms.RandSpatialCropd(keys=["image", "label","coarse_label","ref_label"], roi_size=patch_size,allow_missing_keys = am,
#                                         # max_roi_size = patch_size,
#                                         random_size=False),
#             # transforms.CenterSpatialCropd(keys=["image", "label","coarse_label"], roi_size=patch_size),
#             transforms.SpatialPadd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, spatial_size=patch_size),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#             transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, allow_missing_keys = am, channel_wise=True),
#             transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
#             transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
#             transforms.ToTensord(keys=["image", "label","coarse_label","ref_label"],allow_missing_keys = am,dtype=torch.float),
#         ])
#     return tr_transforms

def transform_train_oss(patch_size = (32,32,32)):
    am = True
    tr_transforms =transforms.Compose([
            transforms.CropForegroundd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, source_key="ref_label", margin=32 if args.patch_size==64 else 8),

            # transforms.RandSpatialCropd(keys=["image", "label","coarse_label","ref_label","dilate_label","guided_image"], roi_size=patch_size,allow_missing_keys = am,
            #                             # max_roi_size = patch_size,
            #                             random_size=False),
            transforms.CenterSpatialCropd(keys=["image", "label","coarse_label","ref_label","dilate_label","guided_image"], allow_missing_keys = am, roi_size=patch_size),
            transforms.SpatialPadd(keys=["image", "label","coarse_label","ref_label","dilate_label","guided_image"], allow_missing_keys = am, spatial_size=patch_size),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, allow_missing_keys = am, channel_wise=True),
            transforms.RandFlipd(["image", "label","coarse_label","ref_label","dilate_label","guided_image"],spatial_axis=2,allow_missing_keys=True, prob=0.5),
            transforms.RandFlipd(["image", "label","coarse_label","ref_label","dilate_label","guided_image"],spatial_axis=1,allow_missing_keys=True, prob=0.5),
            transforms.RandFlipd(["image", "label","coarse_label","ref_label","dilate_label","guided_image"],spatial_axis=0,allow_missing_keys=True, prob=0.5),
            transforms.ToTensord(keys=["image", "label","coarse_label","ref_label","dilate_label","guided_image"],allow_missing_keys = am,dtype=torch.float),
        ])
    return tr_transforms

def transform_val_oss(patch_size = (32,32,32)):
    am = True
    val_transform = transforms.Compose(
            [   
                transforms.CropForegroundd(keys=["image", "label","coarse_label","ref_label"], source_key="ref_label", allow_missing_keys = am, margin=32 if args.patch_size==64 else 8),
                transforms.CenterSpatialCropd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, roi_size=patch_size),
                transforms.SpatialPadd(keys=["image", "label","coarse_label","ref_label"], allow_missing_keys = am, spatial_size=patch_size),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, allow_missing_keys = am, channel_wise=True),
                transforms.ToTensord(keys=["image", "label", "coarse_label","ref_label"],allow_missing_keys = am, dtype=torch.float),
            ]
        )
    return val_transform
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
post_label = AsDiscrete(to_onehot=config.num_cls)
post_pred = AsDiscrete(argmax=True, to_onehot=config.num_cls)
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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'/media/HDD/fanlinqian/work_dirs_othermodels/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)
    shutil.copy('/home/suk/3dOssSeg/code/train_othermodels_oss.py', snapshot_path+'/train.py')
    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train_loader = make_loader(args.split_labeled, transforms_tr = transform_train_oss(patch_size), task = args.task)
    eval_loader = make_loader(args.split_eval, task=args.task, is_training=False, transforms_tr = transform_val_oss(patch_size))
    logging.info(f'{len(train_loader)} itertations per epoch (labeled)')

        # make model, optimizer, and lr scheduler
    model, optimizer = make_model_all()
    dice_loss = DiceCELoss(
        include_background = False, softmax = True, to_onehot_y = True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
    )
    
    scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=50, max_epochs=args.max_epoch
        )
    if args.model_name == "diffunet":
        import torch.nn as nn
        dice_loss = DiceLoss(sigmoid=True)
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.max_epoch//10, max_epochs=args.max_epoch
        )
    # scheduler = ConstantWarmupCosineAnnealingLR(optimizer, warmup_epochs=int(args.max_epoch * 1/3), total_epochs=args.max_epoch, eta_min=5e-6)
    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    best_dice = 0.0
    best_epoch = 0
    max_iters = args.max_epoch * len(train_loader)
    iter_step = [int(max_iters * 2/3),int(max_iters * 5/6 )]
    iter_save = np.arange(0, max_iters, max_iters//10)
    curr_iter = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        dice_list = []
        dice_train_list = []
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            image, label = fetch_data(batch)
            
            if args.model_name == "diffunet":
                x_start = label
                x_start = (x_start) * 2 - 1
                x_t, t, noise = model(x=x_start, pred_type="q_sample")
                preds = model(x=x_t, step=t, image=image, pred_type="denoise")
                loss_dice = dice_loss(preds, label)
                loss_bce = bce_loss(preds, label)

                loss_mse = mse_loss(torch.sigmoid(preds), label)
                loss = loss_dice + loss_bce + loss_mse
                label = torch.argmax(label,dim=1,keepdim=True)
            else:
                label = label.long()
                preds = model(image)
                # from IPython import embed;embed()
                loss = dice_loss(preds, label)
            loss.backward()
            optimizer.step()
            # from IPython import embed;embed()
            curr_iter += 1
            # if curr_iter in iter_step:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.5
            if curr_iter in iter_save:
                save_path = os.path.join(snapshot_path, f'ckpts/model_iter{curr_iter}.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving model to {save_path}')
            
            dice_train = cal_dice(preds, label)
            dice_train_list.append(dice_train.item())
            loss_list.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        if args.need_val:
            model.eval()
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, label = fetch_data(batch)
                    if args.model_name == "diffunet":
                        preds = model(image,pred_type="ddim_sample")
                        label = torch.argmax(label,dim=1,keepdim=True)
                    else:
                        label = label.long()
                        preds = model(image)
                    dice = cal_dice(preds, label)
                    dice_list.append(dice.item())

            dice_mean = np.mean(dice_list)
            writer.add_scalar('evaluate/dice_test', dice_mean, epoch_num)
            
            if dice_mean > best_dice:
                best_dice = dice_mean
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best dice is {best_dice} in epoch {best_epoch}')
        
        dice_train_mean = np.mean(dice_train_list)
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('evaluate/dice_train', dice_train_mean, epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | dice_train: {dice_train_mean}')
        # optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        # if epoch_num - best_epoch == 100:
        #     logging.info(f'Early stop.')
        #     break

    save_path = os.path.join(snapshot_path, f'ckpts/model_iter{curr_iter}.pth')
    torch.save({
        'state_dict': model.state_dict(),
    }, save_path)
    logging.info(f'saving final model to {save_path}')
    writer.close()