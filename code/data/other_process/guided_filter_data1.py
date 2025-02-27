import SimpleITK as sitk
import numpy as np
import scipy.ndimage
from skimage.filters import threshold_otsu  # Otsu's thresholding method

def apply_guided_filter(image, guidance_image, radius=5, epsilon=1e-6):
    """
    应用三维导向滤波 (Guided Filter)
    
    :param image: 输入图像 (3D numpy array)
    :param guidance_image: 引导图像 (3D numpy array)
    :param radius: 滤波的半径 (默认为5)
    :param epsilon: 正则化参数 (默认为1e-6)
    :return: 滤波后的图像
    """
    # 计算图像的均值和方差
    mean_guidance = scipy.ndimage.uniform_filter(guidance_image, size=radius)
    mean_image = scipy.ndimage.uniform_filter(image, size=radius)
    mean_guidance_image = scipy.ndimage.uniform_filter(guidance_image * image, size=radius)
    
    # 计算协方差和方差
    covar = mean_guidance_image - mean_guidance * mean_image
    var_guidance = scipy.ndimage.uniform_filter(guidance_image * guidance_image, size=radius) - mean_guidance ** 2
    
    # 计算a和b系数
    a = covar / (var_guidance + epsilon)
    b = mean_image - a * mean_guidance
    
    # 计算输出
    mean_a = scipy.ndimage.uniform_filter(a, size=radius)
    mean_b = scipy.ndimage.uniform_filter(b, size=radius)
    
    output = mean_a * guidance_image + mean_b
    return output

def load_nii(file_path):
    """
    从文件加载 .nii.gz 数据
    """
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    return img_array

def save_nii(image_array, output_path):
    """
    将处理后的数据保存为 .nii.gz 文件
    """
    img = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(img, output_path)

def binarize_image(image, method="otsu"):
    """
    对图像进行二值化处理
    
    :param image: 输入的图像（3D numpy 数组）
    :param method: 二值化方法，"otsu" 或者自定义阈值
    :return: 二值化后的图像
    """
    if method == "otsu":
        # 使用 Otsu 方法自动计算阈值
        threshold_value = threshold_otsu(image) - 0.01
        binary_image = image > threshold_value
    else:
        # 使用自定义阈值
        threshold_value = method
        binary_image = image > threshold_value
    
    return binary_image.astype(np.uint8)  # 转换为二值化的类型（0和1）
if __name__ == "__main__":
    # 加载两个nii.gz文件
    ids = np.loadtxt('/media/HDD/fanlinqian/ossicular/data1_process/split_txts/complete_DG.txt',
                    dtype=str).tolist()
    from tqdm import tqdm 
    filter_type = "label"
    for id in tqdm(ids):
        path = "/media/HDD/fanlinqian/work_dirs_refiner/Exp_refiner_OSS/20241203-1733198416-diffusion_refiner_oss_denoised/"
        image1 = load_nii(path+f"predictions/{id}_image.nii.gz")
        image2 = load_nii(path+f"predictions/{id}_{filter_type}.nii.gz")

        # 确保两个图像的维度相同
        assert image1.shape == image2.shape, "输入的两个图像尺寸不一致！"

        # 应用三维导向滤波处理
        # filtered_image1 = apply_guided_filter(image1, image2)
        filtered_image2 = apply_guided_filter(image2, image1)
        binary_filtered_image2 = binarize_image(filtered_image2, method="otsu")

        # 保存处理后的结果
        save_nii(filtered_image2, path+f"guided_filter/{id}_{filter_type}_guided.nii.gz")
        save_nii(binary_filtered_image2, path+f"guided_filter/{id}_{filter_type}_guided_filtered.nii.gz")

    print("三维导向滤波处理完成！")
