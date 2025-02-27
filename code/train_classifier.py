"""
bash train_refiner.sh -c 3 -s 6 -p 64 -e classifier -t oss2 -l 1e-2 -n 2000 -d true
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
# from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from data.data_loaders import DatasetOss12
from monai import transforms
from utils import maybe_mkdir,seed_worker
def transform_train_oss(patch_size = (64,64,64)):
    tr_transforms =transforms.Compose([
            transforms.CropForeground(margin=32),
            transforms.CenterSpatialCrop(roi_size=patch_size),
            transforms.SpatialPad(spatial_size=patch_size),
            transforms.RandFlip(spatial_axis=2,prob=0.5),
            transforms.RandFlip(spatial_axis=1,prob=0.5),
            transforms.RandFlip(spatial_axis=0,prob=0.5),
            transforms.ToTensor(dtype=torch.float),
        ])
    return tr_transforms

def transform_val_oss(patch_size = (64,64,64)):
    tr_transforms =transforms.Compose([
            transforms.CropForeground(margin=32),
            transforms.CenterSpatialCrop(roi_size=patch_size),
            transforms.SpatialPad(spatial_size=patch_size),
            transforms.ToTensor(dtype=torch.float),
        ])
    return tr_transforms


    
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
    snapshot_path = f'/media/HDD/fanlinqian/work_dirs_classifier/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)
    shutil.copy('/home/suk/3dOssSeg/code/train_classifier.py', snapshot_path+'/train.py')
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
    
    train_dataset = DatasetOss12(split=args.split_labeled,task=args.task,transform=transform_train_oss())
    val_dataset = DatasetOss12(split=args.split_eval,task=args.task,transform=transform_val_oss())
    train_loader = DataLoader(train_dataset,
                                batch_size = 16, 
                                shuffle=True, 
                                num_workers=args.num_workers,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                drop_last=True,)
    val_loader = DataLoader(val_dataset,
                                batch_size = 8, 
                                num_workers=2,
                                )
    
    logging.info(f'{len(train_loader)} itertations per epoch (labeled)')
    
    from Othermodels.resnet18 import ResNet3D_18
    model = ResNet3D_18(num_classes=1).cuda()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-4)

    for epoch in range(args.max_epoch + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # 前向传播
            outputs = model(inputs).squeeze()
            # print(outputs.shape)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # _, predicted = torch.max(outputs, 1)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        logging.info(f"Epoch [{epoch+1}/{args.max_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        writer.add_scalar('loss/loss', epoch_loss, epoch)
        writer.add_scalar('evaluate/train', epoch_accuracy, epoch)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs).squeeze()
                # _, predicted = torch.max(outputs, 1)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            epoch_accuracy = 100 * correct / total
            logging.info(f"Eval:{epoch_accuracy},{torch.sigmoid(outputs)}")
            writer.add_scalar('evaluate/eval', epoch_accuracy, epoch)
        if epoch%5==0:
            save_path = os.path.join(snapshot_path, f'ckpts/model_{epoch}.pth')
            torch.save({
                    'state_dict': model.state_dict(),
            }, save_path)
            logging.info(f'saving model to {save_path}')
            