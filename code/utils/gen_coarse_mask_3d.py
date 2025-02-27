import os
import numpy as np
import random
import math
import json
import multiprocessing as mp
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import generate_binary_structure

def get_random_structure(size):
    
    choice = np.random.randint(2, 4)
    
    if choice == 1:
        # 3D Rectangular structure
        return np.ones((size, size, size), dtype=np.uint8)
    elif choice == 2:
        # 3D Ellipsoidal structure (approximated)
        x, y, z = np.ogrid[-size//2+1:size//2+1, -size//2+1:size//2+1, -size//2+1:size//2+1]
        mask = x**2 + y**2 + z**2 <= (size//2)**2
        return mask.astype(np.uint8)
    elif choice == 3:
        # Ellipsoidal structure with different size in one dimension
        x, y, z = np.ogrid[-size//2+1:size//2+1, -size//4:size//4, -size//2+1:size//2+1]
        mask = x**2 + (2*y)**2 + z**2 <= (size//2)**2
        return mask.astype(np.uint8)
    elif choice == 4:
        # Ellipsoidal structure with different size in another dimension
        x, y, z = np.ogrid[-size//4:size//4, -size//2+1:size//2+1, -size//2+1:size//2+1]
        mask = (2*x)**2 + y**2 + z**2 <= (size//2)**2
        return mask.astype(np.uint8)
    else:
        raise KeyError
import random
def random_dilate(seg, min=5, max=10):
    # size = random.sample([3,5,7],1)[0]
    # size = 3
    size = random.sample([3,5],1)[0]
    kernel = get_random_structure(size)
    seg = binary_dilation(seg, structure=kernel)
    return seg

def random_erode(seg, min=5, max=10):
    size = random.sample([3,5],1)[0]
    # size = 3
    kernel = get_random_structure(size)
    seg = binary_erosion(seg, structure=kernel,mask=seg)
    return seg

def compute_iou(seg, gt):
    intersection = np.logical_and(seg, gt)
    union = np.logical_or(seg, gt)
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=(0.3,0.7)):
    seg = gt.copy()
    w,h,d = seg.shape
    # for _ in range(250*32): # for LA
    for i in range(1000): # for oss
        lx, ly, lz = np.random.randint(w), np.random.randint(h), np.random.randint(d)
        lw, lh, ld = np.random.randint(lx+1, w+1), np.random.randint(ly+1, h+1), np.random.randint(lz+1, d+1)
        
        # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
        if np.random.rand() < 0.5:
            cx = (lx + lw) // 2
            cy = (ly + lh) // 2
            cz = (lz + ld) // 2
            # if seg[cx, cy, cz] == 0:
            #     seg[cx, cy, cz] = np.random.randint(2)
            seg[cx, cy, cz] = np.random.randint(2)

        if np.random.rand() < 0.3:
            seg[lx:lw, ly:lh, lz:ld] = random_dilate(seg[lx:lw, ly:lh, lz:ld])
        else:
            seg[lx:lw, ly:lh, lz:ld] = random_erode(seg[lx:lw, ly:lh, lz:ld])
        
        if iou_target[0] < compute_iou(seg, gt) < iou_target[1]:
            # print(i,compute_iou(seg, gt))
            break
    # if compute_iou(seg, gt) < 0.5:
    #     seg = random_dilate(seg)
    
    return seg


def perturb_seg_erode(gt, iou_target=(0,0.7)):
    seg = gt.copy()
    w,h,d = seg.shape
    # for _ in range(250*32): # for LA
    for i in range(50): # for oss
        lx, ly, lz = np.random.randint(w), np.random.randint(h), np.random.randint(d)
        lw, lh, ld = np.random.randint(lx+1, w+1), np.random.randint(ly+1, h+1), np.random.randint(lz+1, d+1)
        
        # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
        # if np.random.rand() < 0.5:
        #     cx = (lx + lw) // 2
        #     cy = (ly + lh) // 2
        #     cz = (lz + ld) // 2
        #     if seg[cx, cy, cz] == 1:
        #         seg[cx, cy, cz] = np.random.randint(2)
            # seg[cx, cy, cz] = np.random.randint(2)

        # if np.random.rand() < 0.15:
        #     seg[lx:lw, ly:lh, lz:ld] = random_dilate(seg[lx:lw, ly:lh, lz:ld])
        # else:
        #     seg[lx:lw, ly:lh, lz:ld] = random_erode(seg[lx:lw, ly:lh, lz:ld])
        seg[lx:lw, ly:lh, lz:ld] = random_erode(seg[lx:lw, ly:lh, lz:ld])        
        if iou_target[0] < compute_iou(seg, gt) < iou_target[1]:
            # print(i,compute_iou(seg, gt))
            break
    # if compute_iou(seg, gt) < 0.5:
    #     seg = random_dilate(seg)
    return seg

def perturb_seg_dilate(gt, iou_target=(0,0.7)):
    seg = gt.copy()
    w,h,d = seg.shape
    # for _ in range(250*32): # for LA
    for i in range(50): # for oss
        lx, ly, lz = np.random.randint(w), np.random.randint(h), np.random.randint(d)
        lw, lh, ld = np.random.randint(lx+1, w+1), np.random.randint(ly+1, h+1), np.random.randint(lz+1, d+1)
        
        # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
        if np.random.rand() < 0.5:
            cx = (lx + lw) // 2
            cy = (ly + lh) // 2
            cz = (lz + ld) // 2
            if seg[cx, cy, cz] == 0:
                seg[cx, cy, cz] = np.random.randint(2)
            # seg[cx, cy, cz] = np.random.randint(2)

        # if np.random.rand() < 0.15:
        #     seg[lx:lw, ly:lh, lz:ld] = random_dilate(seg[lx:lw, ly:lh, lz:ld])
        # else:
        #     seg[lx:lw, ly:lh, lz:ld] = random_erode(seg[lx:lw, ly:lh, lz:ld])
        seg[lx:lw, ly:lh, lz:ld] = random_dilate(seg[lx:lw, ly:lh, lz:ld])        
        if iou_target[0] < compute_iou(seg, gt) < iou_target[1]:
            # print(i,compute_iou(seg, gt))
            break
    # if compute_iou(seg, gt) < 0.5:
    #     seg = random_dilate(seg)
    return seg

def perturb_seg_erode_all(gt):
    seg = gt.copy()
    w,h,d = seg.shape
    seg_mid = []
    for i in range(6):
        ones_indices = np.argwhere(seg == 1)
        # print("i=",i,len(ones_indices))
        selected_indices = ones_indices[np.random.choice(len(ones_indices), 1, replace=False)]
        for idx in selected_indices:
            cx, cy, cz = idx
            # lx, ly, lz = np.random.randint(cx), np.random.randint(cy), np.random.randint(cz)
            # rx, ry, rz = np.random.randint(cx+1,w+1), np.random.randint(cy+1,h+1), np.random.randint(cz+1,d+1)
            wid = 8
            lx, ly, lz = np.random.randint(max(0,cx-wid),cx), np.random.randint(max(0,cy-wid),cy), np.random.randint(max(0,cz-wid),cz)
            rx, ry, rz = np.random.randint(cx+1,min(w+1,cx+wid+1)), np.random.randint(cy+1,min(h+1,cy+wid+1)), np.random.randint(cz+1,min(d+1,cz+wid+1))
            seg[cx, cy, cz] = 0
            seg_tmp = seg[lx:rx, ly:ry, lz:rz]
            seg[lx:rx, ly:ry, lz:rz] = random_erode(seg_tmp)
        seg_mid.append(seg.copy())
    # print(len(seg_mid),seg_mid[0].shape)
    return seg_mid

def perturb_seg_erode_mid(gt,prob):
    seg = gt.copy()
    w,h,d = seg.shape
    orig_one = gt.sum()
    while(seg.sum()/orig_one > prob+0.05):
        ones_indices = np.argwhere(seg == 1)
        # print("i=",i,len(ones_indices))
        
        selected_indices = ones_indices[np.random.choice(len(ones_indices), 1, replace=False)]
        # for idx in selected_indices:
        idx = selected_indices[0]
        cx, cy, cz = idx
        # lx, ly, lz = np.random.randint(cx), np.random.randint(cy), np.random.randint(cz)
        # rx, ry, rz = np.random.randint(cx+1,w+1), np.random.randint(cy+1,h+1), np.random.randint(cz+1,d+1)
        wid = 8
        try:
            lx, ly, lz = np.random.randint(max(0,cx-wid),cx), np.random.randint(max(0,cy-wid),cy), np.random.randint(max(0,cz-wid),cz)
            rx, ry, rz = np.random.randint(cx+1,min(w+1,cx+wid+1)), np.random.randint(cy+1,min(h+1,cy+wid+1)), np.random.randint(cz+1,min(d+1,cz+wid+1))
        except Exception:
            continue
        seg[cx, cy, cz] = 0
        seg_tmp = seg[lx:rx, ly:ry, lz:rz]
        seg[lx:rx, ly:ry, lz:rz] = random_erode(seg_tmp)
    # print(prob,seg.sum()/orig_one)
    return seg

def perturb_seg_erode_mid_slice(gt,prob):
    seg = gt.copy()
    w,h,d = seg.shape
    orig_one = gt.sum()
    
    slices = [gt.take(i, axis=1) for i in range(gt.shape[1])]
    print(seg.sum()/orig_one)
    for ii in range(6):
        ss = 0
        for i,slice_2d in enumerate(slices):
            erode_slice = binary_erosion(slice_2d,iterations=1)
            slices[i] = erode_slice
            ss += erode_slice.sum()
        print(ss/orig_one)
    
    exit(0)
    
# def perturb_seg_dilate(gt, iou_target=(0,0.6)):
#     def random_dilate_pri(seg, min=5, max=10):
#         # size = random.sample([3,5,7],1)[0]
#         # size = 3
#         size = random.sample([3,5],1)[0]
#         kernel = get_random_structure(size)
#         seg = binary_dilation(seg, structure=kernel)
#         return seg
#     seg = gt.copy()
#     w,h,d = seg.shape
#     # for _ in range(250*32): # for LA
#     for i in range(1000): # for oss
#         lx, ly, lz = np.random.randint(w), np.random.randint(h), np.random.randint(d)
#         lw, lh, ld = np.random.randint(lx+1, w+1), np.random.randint(ly+1, h+1), np.random.randint(lz+1, d+1)
        
#         # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
#         if np.random.rand() < 0.1:
#             cx = (lx + lw) // 2
#             cy = (ly + lh) // 2
#             cz = (lz + ld) // 2
#             # if seg[cx, cy, cz] == 0:
#             #     seg[cx, cy, cz] = np.random.randint(2)
#             seg[cx, cy, cz] = np.random.randint(2)

#         if np.random.rand() > 0.05:
#             seg[lx:lw, ly:lh, lz:ld] = random_dilate_pri(seg[lx:lw, ly:lh, lz:ld])
#         else:
#             seg[lx:lw, ly:lh, lz:ld] = random_erode(seg[lx:lw, ly:lh, lz:ld])
        
#         if iou_target[0] < compute_iou(seg, gt) < iou_target[1]:
#             # print(i,compute_iou(seg, gt))
#             break
#     # if compute_iou(seg, gt) < 0.5:
#     #     seg = random_dilate(seg)
#     return seg



# def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target=0.8):
#     # 复制图像以进行修改
#     modified_image = image.copy()

#     # 获取图像的形状
#     depth, height, width = image.shape

#     # 随机选择一些点进行修改
#     num_points = int(depth * height * width * sample_rate)
#     for _ in range(num_points):
#         dz, dy, dx = np.random.randint(0, depth), np.random.randint(0, height), np.random.randint(0, width)
        
#         # 计算与中心的距离
#         center = np.array([depth, height, width]) // 2
#         displacement = np.array([dz, dy, dx]) - center
        
#         # 根据move_rate调整点的位置
#         if np.random.rand() < move_rate:
#             dz += int(displacement[0] * move_rate)
#             dy += int(displacement[1] * move_rate)
#             dx += int(displacement[2] * move_rate)
        
#         # 确保索引在有效范围内
#         dz = np.clip(dz, 0, depth - 1)
#         dy = np.clip(dy, 0, height - 1)
#         dx = np.clip(dx, 0, width - 1)
        
#         # 修改点的值为0或1
#         modified_image[dz, dy, dx] = np.random.randint(2)

#     # 使用扰动函数进一步修改图像
#     modified_image = perturb_seg(modified_image, iou_target)

#     return modified_image
import matplotlib.pyplot as plt
def plot_3d(data,save_path):
    
# 检查数据的形状
    print(data.shape)

    # 创建一个 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据的维度
    x, y, z = data.nonzero()  # 获取所有非零点的坐标
    values = data[x, y, z]    # 获取对应的值

    # 绘制散点图或表面图
    ax.scatter(x, y, z, c=values, cmap='viridis')  # 使用散点图，颜色映射

    # 设置标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 显示图像
    plt.show()
    plt.savefig(save_path)
def run_inst(img_path):
    # 读取 3D 图像的代码需要根据具体格式调整
    gt_mask = np.load(img_path)  # 假设为 .npy 文件
    assert gt_mask is not None
    # plot_3d(gt_mask,img_path.replace('_label.npy','_label.jpg'))
    coarse_mask = perturb_seg(gt_mask)
    # plot_3d(coarse_mask,img_path.replace('_label.npy','_coarse_label.jpg'))
    out = sitk.GetImageFromArray(coarse_mask.astype(np.float32))
    sitk.WriteImage(out,img_path.replace('_label.npy','_coarse_label.nii.gz'))  # 保存为 .npy 文件

if __name__ == '__main__':
    import SimpleITK as sitk
    label_files = [f for f in os.listdir('/media/HDD/fanlinqian/LASeg/LA_process/npy/') if f.endswith('_label.npy')]
    # print(label_files)
    for p in tqdm(label_files):
        run_inst('/media/HDD/fanlinqian/LASeg/LA_process/npy/' + p)