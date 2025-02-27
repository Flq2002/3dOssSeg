"""
bash train_refiner.sh -c 5 -s 6 -p 64 -e casca -t oss1 -l 5e-5 -n 2000 -d true -b new_manual1
bash train_refiner.sh -c 5 -s 4 -p 64 -e diffusion_refiner_oss -t oss1 -l 5e-5 -n 1000 -d true -b /media/HDD/fanlinqian/work_dirs_othermodels/Exp_OSS/20241105-1730777670-swin_unetr/predictions/
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
# from Segrefiner.segrefiner_base import SegRefiner as Seg   efiner_oss
from utils import EMA, maybe_mkdir, get_lr, fetch_data_oss, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetOss,DatasetOss_noise
from utils.config import Config
from utils import read_list,read_data_coarse
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR,ConstantWarmupCosineAnnealingLR
from cascadpsp.models.psp.pspnet import PSPNet
from cascadpsp.models.sobel_op import SobelComputer
from cascadpsp.util.metrics_compute import compute_loss_and_metrics,iou_hooks_to_be_used
from cascadpsp.util.logger import BoardLogger
from cascadpsp.util.log_integrator import Integrator
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
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50').cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-4)

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

para = {
    "ce_weight":[0.0, 1.0, 0.5, 1.0, 1.0, 0.5],
    "l1_weight":[1.0, 0.0, 0.25, 0.0, 0.0, 0.25],
    "l2_weight":[1.0, 0.0, 0.25, 0.0, 0.0, 0.25],
    "grad_weight":5.
}
def cal_dice(target, mask, eps=1e-3):
    target = target.clone().detach() >= 0.5
    mask = mask.clone().detach() >= 0.5
    si = (target & mask).sum(-1).sum(-1).sum(-1)
    return (2 * si) / (target.sum(-1).sum(-1).sum(-1) + mask.sum(-1).sum(-1).sum(-1) + eps)
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
    snapshot_path = f'/media/HDD/fanlinqian/work_dirs_casca/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)
    shutil.copy('/home/suk/3dOssSeg/code/train_casca.py', snapshot_path+'/train.py')
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


    # scheduler = LinearWarmupCosineAnnealingLR(optimizer,args.max_epoch * 0.05,args.max_epoch)
    # scheduler = ConstantWarmupCosineAnnealingLR(optimizer, warmup_epochs=int(args.max_epoch * 5/6), total_epochs=args.max_epoch, eta_min=5e-6)
    
    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    best_dice = 0.0
    best_epoch = 0
    max_iters = args.max_epoch * len(train_loader)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[22500, 37500],0.1)
    # iter_step = [int(max_iters * 2/3),int(max_iters * 5/6 ),int(max_iters * 11/12 )]
    iter_save = np.arange(0, max_iters, max_iters//10)
    sobel_compute = SobelComputer()
    curr_iter = 0
    logger = BoardLogger(args.exp)
    train_integrator = Integrator(logger)
    train_integrator.add_hook(iou_hooks_to_be_used)
    report_interval=50
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
        # for batch in tqdm(train_loader):
        tot_dice =0
        for batch in train_loader:
            # from IPython import embed;embed()
            optimizer.zero_grad()
            im, gt, seg = fetch_data_oss(batch)

            images = model(im, seg)
            images['im'] = im
            images['seg'] = seg
            images['gt'] = gt
            sobel_compute.compute_edges(images)
            loss_and_metrics = compute_loss_and_metrics(images, para)
            (loss_and_metrics['total_loss']).backward()
            optimizer.step()

            tmp_dice = cal_dice(gt,images["pred_224"])
            tot_dice+=tmp_dice.item()
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
            
            if scheduler is not None:
                scheduler.step()
            train_integrator.add_dict(loss_and_metrics)

        train_integrator.finalize('train', epoch_num)
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss', train_integrator.values["total_loss"]/len(train_loader), epoch_num)
        writer.add_scalar('iou/orig', train_integrator.values["iou/orig_iou"], epoch_num)
        writer.add_scalar('iou/new', train_integrator.values["iou/new_iou_224"], epoch_num)
        logging.info(f'epoch: {epoch_num} loss: {train_integrator.values["total_loss"]}, test_iou : {train_integrator.values["iou/new_iou_224"]}, "dice": {tot_dice/len(train_loader)}')
        writer.add_scalar('dice/new', tot_dice/len(train_loader), epoch_num)
        train_integrator.reset_except_hooks()
        
    save_path = os.path.join(snapshot_path, f'ckpts/final_model.pth')
    torch.save({
        'state_dict': model.state_dict(),
    }, save_path)
    logging.info(f'saving final model to {save_path}')
    writer.close()
