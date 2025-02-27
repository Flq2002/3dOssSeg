"""
bash train_refiner.sh -c 4 -s 6 -p 64 -e diffusion_refiner_oss -t oss1 -l 5e-5 -n 3000 -d true -b new_manual1
bash train_refiner.sh -c 5 -s 4 -p 64 -e diffusion_refiner_oss -t oss1 -l 5e-5 -n 3000 -d true -b /media/HDD/fanlinqian/work_dirs_othermodels/Exp_OSS/20241105-1730777670-swin_unetr/predictions/
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
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--timestep', type=int, default=6)
parser.add_argument('--need_val', action='store_true', default=False)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--base_model', type=str, default=None)
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
from Segrefiner.segrefiner_base import SegRefiner as Segefiner_oss
from utils import EMA, maybe_mkdir, get_lr, fetch_data_oss, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup
# from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetOss,DatasetOss_noise
from utils.config import Config
from utils import read_list,read_data_coarse
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR,ConstantWarmupCosineAnnealingLR
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
            is_coarse = "random_0",
            # is_coarse = "false",
            base_model=args.base_model,
            data_type='data2' if args.task == 'oss2' else 'data1',
            # diffusion_cfg=diffusion_cfg
        )
        return DataLoader(
            dst,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            # num_workers = 1,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True,
            
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms_tr,
            # is_coarse = 'noise',
            is_coarse = "flase",
            data_type='data2' if args.task == 'oss2' else 'data1',
            base_model=args.base_model
        )
        return DataLoader(dst, pin_memory=True,num_workers=2)

def make_model_all():
    model = SegRefiner_oss(
        diffusion_cfg = diffusion_cfg,
        denoise_cfg = denoise_cfg

    ).cuda()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    return model, optimizer



from monai import transforms
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
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            transforms.ToTensord(keys=["image", "label","coarse_label","ref_label","dilate_label","guided_image"],allow_missing_keys = am,dtype=torch.float),
        ])
    return tr_transforms

def transform_val_oss(patch_size = (32,32,32)):
    val_transform = transforms.Compose(
        [   
            transforms.CropForegroundd(keys=["image", "label","coarse_label","ref_label"], source_key="ref_label", margin=32 if args.patch_size==64 else 8),
            transforms.CenterSpatialCropd(keys=["image", "label","coarse_label"], roi_size=patch_size),
            transforms.SpatialPadd(keys=["image", "label","coarse_label"], spatial_size=patch_size),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "coarse_label"],dtype=torch.float),
        ]
    )
    return val_transform
def transform_val_oss_other(patch_size = (32,32,32)):
    val_transform = transforms.Compose(
        [   
            transforms.CropForegroundd(keys=["image", "label","ref_label"], source_key="ref_label", margin=32 if args.patch_size==64 else 8),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "coarse_label"],dtype=torch.float),
        ]
    )
    return val_transform

diffusion_cfg=dict(
    betas=dict(
        type='linear',
        start=0.8,
        stop=0,
        # start=0.9,
        # stop=0.1,
        num_timesteps=args.timestep),
    diff_iter=False)
denoise_cfg=dict(
    # in_channels=2+3,
    in_channels=1,
    # in_channels=2,
    out_channels=1,
    model_channels=128,
    # model_channels=64,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    attention_strides=(16, 32),
    learn_time_embd=True,
    channel_mult = (1,1,2,2,4,4),
    # dropout=0.0,
    dropout=0.3,
    dims = 3,
    num_timesteps=args.timestep,
    use_img_embd = 'layers_orig',
    # use_img_embd = None,
    )

import SimpleITK as sitk
def loss_diff_func(data):
    "B,C,H,W,D"
    diffH = data[:,:,:-2,...] + data[:,:,2:,...] - 2*data[:,:,1:-1,...]
    diffW = data[:,:,:,:-2,...] + data[:,:,:,2:,...] - 2*data[:,:,:,1:-1,...]
    diffD = data[...,:-2] + data[...,2:] - 2*data[...,1:-1]
    return (diffH.abs().mean()+diffD.abs().mean()+diffW.abs().mean())
    # return (diffH.mean()+diffD.mean()+diffW.mean()) 
def loss_guided_func(pred,guided):
    pred = pred[guided!=0]
    guided = guided[guided!=0].sigmoid()
    return (pred*torch.log(pred/guided)).mean()
    
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
    snapshot_path = f'/media/HDD/fanlinqian/work_dirs_refiner/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)
    shutil.copy('/home/suk/3dOssSeg/code/train_diffusion_refiner_oss.py', snapshot_path+'/train.py')
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
    if args.need_val:
        if args.base_model is None or "manual" in args.base_model:
            eval_loader = make_loader(args.split_eval, task=args.task, is_training=False, transforms_tr = transform_val_oss(patch_size))
        else:
            eval_loader = make_loader(args.split_eval, task=args.task, is_training=False, transforms_tr = transform_val_oss_other(patch_size))
    logging.info(f'{len(train_loader)} itertations per epoch (labeled)')

    # make model, optimizer, and lr scheduler
    model, optimizer = make_model_all()
    logging.info(denoise_cfg)


    # prepare other losses
    from Segrefiner.losses.cross_gradient_loss import CrossGradientLoss
    loss_gradient_func = CrossGradientLoss()
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer,args.max_epoch * 0.05,args.max_epoch)
    # scheduler = ConstantWarmupCosineAnnealingLR(optimizer, warmup_epochs=int(args.max_epoch * 5/6), total_epochs=args.max_epoch, eta_min=5e-6)
    scheduler = None
    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    best_dice = 0.0
    best_epoch = 0
    max_iters = args.max_epoch * len(train_loader)
    # iter_step = [int(max_iters * 2/3),int(max_iters * 5/6 ),int(max_iters * 11/12 )]
    iter_save = np.arange(0, max_iters, max_iters//10)
    curr_iter = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_mask_list = []
        loss_texture_list = []
        loss_gradient_list = []
        loss_guided_list = []
        loss_diff_list = []
        iou_list = []
        dice_list = []
        loss_cover_list = []
        model.train()
        for batch in tqdm(train_loader):
            # from IPython import embed;embed()
            optimizer.zero_grad()
            image, label, coarse_label = fetch_data_oss(batch)
            # image, label = fetch_data_oss(batch,coarse=False)
            # dilate_label = batch["dilate_label"].cuda()
            # guided_image = batch["guided_image"].cuda()
            label = label.long()
            # coarse_label = coarse_label.long()
            if args.mixed_precision:
                with autocast():
                    loss_map = model.forward_train(image, label, coarse_label)
                    loss = loss_map['loss_mask'] + loss_map['loss_texture'] * 5
                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                loss_map,preds_logit = model.forward_train(image, label, coarse_label)
                # loss_map,preds_logit = model.forward_train(image, label, None, batch["seg_mid"], batch["t"])
                # def save_nii(data,ttt):
                #     out = sitk.GetImageFromArray(data.squeeze().cpu().detach().numpy())
                #     sitk.WriteImage(out,"/media/HDD/fanlinqian/ossicular/tmp/"+ttt)
                # save_nii(preds_logit,"preds_logit.nii.gz")
                # data_id = batch["data_id"]
                # save_nii(guided_image,f"{data_id}_guided_image.nii.gz")
                # save_nii(label.int(),f"{data_id}_label.nii.gz")
                # save_nii(coarse_label.int(),f"{data_id}_coarse_label.nii.gz")
                # save_nii(image,f"{data_id}_image.nii.gz")
                # from IPython import embed;embed()
                # loss_map["loss_gradient"] = loss_gradient_func(preds_logit,image,dilate_label)
                # loss_map["loss_guided"] = loss_guided_func(preds_logit,guided_image*10)
                # loss_map["loss_diff"] = loss_diff_func(preds_logit)
                # loss = loss_map['loss_mask'] + loss_map['loss_texture'] * 5 + loss_map["loss_gradient"] * 500 + loss_map["loss_guided"] * 5 + 5*loss_map['loss_diff']
                loss = loss_map['loss_mask'] + loss_map['loss_texture'] * 5 #+ 0.1*loss_map['loss_cover'] #+ loss_map["loss_guided"]*0.5
                # loss = loss_map['loss_texture']
                loss.backward()
                optimizer.step()

            # max_grad = max(param.grad.max().item() for param in model.parameters() if param.grad is not None)
            # print(max_grad)
            # continue
            
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
                
            loss_list.append(loss.item())
            loss_mask_list.append(loss_map['loss_mask'].item())
            loss_texture_list.append(loss_map['loss_texture'].item())
            # loss_cover_list.append(loss_map['loss_cover'].item())
            # loss_gradient_list.append(loss_map['loss_gradient'].item())
            # loss_guided_list.append(loss_map['loss_guided'].item())
            # loss_diff_list.append(loss_map['loss_diff'].item())
            iou_list.append(loss_map['iou'].item())
            dice_list.append(loss_map['dice'].item())
        # from IPython import embed;embed() 
        iou_mean = np.mean(iou_list)
        dice_mean = np.mean(dice_list)
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss/tot', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/mask', np.mean(loss_mask_list), epoch_num)
        writer.add_scalar('loss/texture', np.mean(loss_texture_list), epoch_num)
        # writer.add_scalar('loss/cover', np.mean(loss_cover_list), epoch_num)
        # writer.add_scalar('loss/gradient', np.mean(loss_gradient_list), epoch_num)
        # writer.add_scalar('loss/guided', np.mean(loss_guided_list), epoch_num)
        # writer.add_scalar('loss/diff', np.mean(loss_diff_list), epoch_num)
        writer.add_scalar('evaluate/iou', iou_mean, epoch_num)
        writer.add_scalar('evaluate/dice', dice_mean, epoch_num)

        

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | train_iou : {iou_mean} | train_dice : {dice_mean} | lr : {get_lr(optimizer)} | loss_mask : {np.mean(loss_mask_list)} | loss_texture : {np.mean(loss_texture_list)}')

        if args.need_val and epoch_num%10==0:
            model.eval()
            test_dice, test_iou = 0.,0.
            with torch.inference_mode():
                for batch in tqdm(eval_loader):
                    image, label, coarse_label = fetch_data_oss(batch)
                    pred = model.single_test(image,coarse_label,patch_size).squeeze()
                    dice,iou=model.cal_dice(label,pred).item(),model.cal_iou(label,pred).item()
                    # from IPython import embed;embed()
                    test_dice += dice
                    test_iou += iou
                test_dice /= len(eval_loader)
                test_iou /= len(eval_loader)
            logging.info(f'epoch {epoch_num} : test_iou : {test_iou} | test_dice : {test_dice}')
            writer.add_scalar('evaluate/test_iou', test_iou, epoch_num)
            writer.add_scalar('evaluate/test_dice', test_dice, epoch_num)
            if test_dice > best_dice:
                best_dice = test_dice
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                logging.info(f'saving best model to {save_path}')
                logging.info(f'\t best dice is {best_dice} in epoch {best_epoch}')

        if scheduler is not None:
            scheduler.step()
        
        # optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        # if epoch_num - best_epoch == config.early_stop_patience:
        #     logging.info(f'Early stop.')
        #     break
    save_path = os.path.join(snapshot_path, f'ckpts/final_model.pth')
    torch.save({
        'state_dict': model.state_dict(),
    }, save_path)
    logging.info(f'saving final model to {save_path}')
    writer.close()
