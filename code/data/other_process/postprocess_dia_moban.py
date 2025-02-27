import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

def load_nifti(file_path):
    """
    加载 nii.gz 文件
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))
def save_nifti(volume, output_path):
    """
    将处理后的三维数据体保存为 .nii.gz 文件
    """
    output_image = sitk.GetImageFromArray(volume.astype(np.uint8))  # 转为 SimpleITK 图像
    sitk.WriteImage(output_image, output_path)

import numpy as np

def dilate_slice(slice):
    template1 = np.zeros((2,2))
    template1[0,0] = 1
    template1[1,1] = 1 
    template2 = 1-template1
    template3 = np.ones((3,2))
    template3[1,0] = 0
    template3[1,1] = 0
    
    dilate_slice = slice.copy()
    
    for i in range(1,slice.shape[0]-2):
        for j in range(1,slice.shape[1]-2):
            if slice[i,j]==0:
                if (slice[i-1:i+2,j:j+2] == template3).all():
                    dilate_slice[i,j] = 1
                if (slice[i-1:i+1,j:j+2] == template1).all():#or (slice[i:i+2,j-1:j+1] == template1).all():
                    dilate_slice[i,j] = 1
                if (slice[i:i+2,j:j+2] == template2).all():# or (slice[i-1:i+1,j-1:j+1] == template2).all():
                    dilate_slice[i,j] = 1
                
    return dilate_slice


def mor_dilate_slice(slice):
    # print("0",slice.sum())
    dilated_slice = dilate_slice(slice)
    # print("1",dilated_slice.sum())
    dilated_slice = binary_closing(dilated_slice, iterations=2)
    # print("2",dilated_slice.sum())
    dilated_slice = dilate_slice(dilated_slice)
    # print("3",dilated_slice.sum())
    # dilated_slice = binary_closing(dilated_slice, iterations=1)
    # print("4",dilated_slice.sum())
    # dilated_slice = dilate_slice(dilated_slice)
    # print("5",dilated_slice.sum())
    return dilated_slice


if __name__ == "__main__":
    from scipy.ndimage import binary_dilation,binary_closing,binary_erosion
    model_name = "20250123-1737619983-diffusion_refiner_oss"
    path = f"/media/HDD/fanlinqian/work_dirs_refiner/Exp_refiner_OSS/{model_name}/predictions/"
    ids = np.loadtxt('/media/HDD/fanlinqian/ossicular/data1_process/split_txts/manual_test.txt',
                        dtype=str).tolist()
    save_path = f"/media/HDD/fanlinqian/work_dirs_refiner/Exp_refiner_OSS/{model_name}/dilate/"
    
    for id in tqdm(ids):
        file_path = path+f"{id}_pred.nii.gz"
        file = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        dilate_file = file.copy()
        
        for j in range(file.shape[1]):
        # for j in range(25,31):
            # dilate_file[:,j,:] = dilate_slice(file[:,j,:])
            # from matplotlib import pyplot as plt
            # plt.imshow(file[:,j,:])
            # plt.savefig(save_path+f"before{j}.png")
            dilate_file[:,j,:] = mor_dilate_slice(file[:,j,:])
            # plt.imshow(dilate_file[:,j,:])
            # plt.savefig(save_path+f"after{j}.png")
        
        save_nifti(dilate_file,save_path+f"{id}.nii.gz")
    # break