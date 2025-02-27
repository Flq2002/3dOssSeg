import numpy as np
from scipy.ndimage import label
import os

def remove_small_components(segmentation, min_size=10):
    labeled_array, num_features = label(segmentation)
    new_segmentation = np.zeros_like(segmentation)

    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        size = np.sum(component)
        # print(size)
        if size >= min_size:
            new_segmentation[component] = 1  # 或者使用i进行标记

    return new_segmentation

def keep_largest_components(segmentation, top_n=3):
    labeled_array, num_features = label(segmentation)
    component_sizes = []

    # 计算每个组件的大小
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        size = np.sum(component)
        component_sizes.append((i, size))

    # 根据大小排序，保留前三个最大的组件
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_components = [comp[0] for comp in component_sizes[:top_n]]
    print(component_sizes)
    # 创建新的分割结果，仅保留最大组件
    new_segmentation = np.zeros_like(segmentation)
    all_sum = 0
    for i in largest_components:
        component = (labeled_array == i)
        new_segmentation[component] = 1  # 或者使用i进行标记
        all_sum += np.sum(component)
        if all_sum > 1000:
            break
    return new_segmentation

# def load_segmentation_from_label(label_name):
#     # 实现加载分割数据的函数，例如使用 np.load
#     return np.load(label_name)
import SimpleITK as sitk
def load_segmentation_from_label(label_name):
    itk_img = sitk.ReadImage(label_name)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr
from tqdm import tqdm
def process_labels_from_file(label_file, min_size=10):
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    

    for label_name in tqdm(labels):
        base_name = label_name+'.nii.gz'
        segmentation = load_segmentation_from_label(file_path+"name2/"+base_name)
        cleaned_segmentation = keep_largest_components(segmentation, 1)
        
        # 构建新文件名并保存
        new_file_name = f"{file_path}filter_label/{label_name}_label_filter.npy"
        # np.save(new_file_name, cleaned_segmentation)
        out = sitk.GetImageFromArray(cleaned_segmentation)
        sitk.WriteImage(out, new_file_name.replace(".npy",".nii.gz"))
# 示例用法
if __name__ == '__main__':
    file_path = '/media/HDD/fanlinqian/ossicular/data2_process/label_manual/'
    process_labels_from_file('/media/HDD/fanlinqian/ossicular/data2_process/split_txts/all.txt', min_size=50)
