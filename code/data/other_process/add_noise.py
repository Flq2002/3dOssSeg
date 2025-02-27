import SimpleITK as sitk
import numpy as np

# 图像归一化到 0-255 并转换为 uint8
def normalize_to_uint8(image_data):
    original_min = np.min(image_data)
    original_max = np.max(image_data)
    normalized = ((image_data - original_min) / (original_max - original_min) * 255).astype(np.uint8)
    return normalized, original_min, original_max

# 反归一化到原始范围
def denormalize_from_uint8(normalized_data, original_min, original_max):
    return (normalized_data.astype(np.float32) / 255) * (original_max - original_min) + original_min

def load_nifti_sitk(file_path):
    print(file_path)
    image = sitk.ReadImage(file_path)
    image_data = sitk.GetArrayFromImage(image)  # 转为 NumPy 数组
    return image_data, image
from scipy.ndimage import gaussian_filter
def compute_gradient(image, sigma=2):
    # 计算图像在每个轴上的一阶梯度
    grad_x = gaussian_filter(image, sigma=sigma, order=[1, 0, 0])
    grad_y = gaussian_filter(image, sigma=sigma, order=[0, 1, 0])
    grad_z = gaussian_filter(image, sigma=sigma, order=[0, 0, 1])
    return np.sqrt(grad_x**2+grad_y**2+grad_z**2)
    # return grad_x
def save_nifti_sitk(data, reference_image, output_path):
    # data = compute_gradient(data)
    output_image = sitk.GetImageFromArray(data)  # 转为 SimpleITK 图像
    if reference_image is not None:
        output_image.CopyInformation(reference_image)  # 复制空间信息
    sitk.WriteImage(output_image, output_path)

import numpy as np
from scipy.ndimage import uniform_filter

def calculate_rms(input_array, patch_size):
    """
    对三维数据的每一个点，计算其周围指定大小patch的RMS。
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size 必须是奇数！")

    # 使用 uniform_filter 计算局部平方的均值
    squared_data = input_array**2
    mean_squared = uniform_filter(squared_data, size=patch_size, mode='constant')
    
    # 计算RMS值
    rms = np.sqrt(mean_squared)
    
    return rms
def try1():
    # clean_orig_image + eps*noise_image_noise
    pass
def try2(orig_image,noise_image,noise):
    orig_rms = calculate_rms(norm_maxmin(orig_image),3)
    noise_rms = calculate_rms(norm_maxmin(noise_image),3)
    ratio = orig_rms/noise_rms
    ratio = norm_maxmin(ratio)
    return noise*ratio
def norm_maxmin(data):
    return (data-data.min())/(data.max()-data.min())

from transform_icp import trans_icp
def try3(source,target,noise):
    source_point = trans_icp.extract_point_cloud(norm_maxmin(source))
    target_point = trans_icp.extract_point_cloud(norm_maxmin(target))
    T = trans_icp.icp(source_point,target_point)
    align = trans_icp.apply_transformation_to_volume(source,T)
    align_noise = trans_icp.apply_transformation_to_volume(noise,T)
    from IPython import embed;embed()
    save_nifti_sitk(align,None,noise_image_path.replace("nii_incomplete","tmp").replace("image","image_transform"))

    return align_noise

def try4(source,target,noise, noise_ref):
    from transform_itk import register_images,apply_transformation
    align, transformation = register_images(target, source)
    save_nifti_sitk(align,None,noise_image_path.replace("nii_incomplete","tmp").replace("image","image_transform"))
    align_noise = apply_transformation(noise, noise_ref, transformation)
    # align_noise = np.where(np.abs(align_noise) > np.abs(noise_ref), align_noise, noise_ref)
    align_noise[align_noise==0] = noise_ref[align_noise==0]
    save_nifti_sitk(align_noise,None,noise_image_path.replace("nii_incomplete","tmp").replace("image","noise_transform_plus2"))
    return align_noise
    

import cv2
def add_noise_to_image_norm(orig_image_path, noise_image_path, output_noise_image_path, eps):
    """
    norm版本
    将噪声按百分比添加到干净图像并保存结果。
    """
    orig_image, reference_data = load_nifti_sitk(orig_image_path)
    noise_image, _ = load_nifti_sitk(noise_image_path)
    orig_image_noise,_ = load_nifti_sitk(orig_image_path.replace("nii_complete","tmp").replace("image","noise"))
    noise_image_noise,_ = load_nifti_sitk(noise_image_path.replace("nii_incomplete","noise_template").replace("image","noise"))

    orig_denoise_image,_ = load_nifti_sitk(orig_image_path.replace("nii_complete","tmp").replace("image","denoised_image"))
    noise_denoise_image,_ = load_nifti_sitk(noise_image_path.replace("nii_incomplete","noise_template").replace("image","denoised_image"))

    norm_orig_image, original_min, original_max = normalize_to_uint8(orig_image)
    norm_clean_orig_image = norm_orig_image-orig_image_noise
    # denorm_clean_orig_image = denormalize_from_uint8(clean_orig_image, original_min, original_max)
    # from IPython import embed;embed()
    try_noise = noise_image_noise
    # try_noise = try2(orig_denoise_image,noise_denoise_image,noise_image_noise)
    # try_noise = try3(noise_denoise_image,orig_denoise_image,noise_image_noise)
    # try_noise = try4(noise_denoise_image,orig_denoise_image,noise_image_noise,orig_image_noise)
    # try_noise = np.where(np.abs(try_noise) < np.abs(orig_image_noise), try_noise, orig_image_noise)
    # try_noise = np.maximum(try_noise,orig_image_noise)
    # try_noise = 0.3*try_noise+0.1*noise_image_noise
    # from denoisy_bi import process_volume_all_directions
    # try_noise = process_volume_all_directions(try_noise)
    denorm_noise = denormalize_from_uint8(norm_clean_orig_image+eps*try_noise, original_min, original_max)
    # from IPython import embed;embed()

    # 转换回 SimpleITK 图像
    save_nifti_sitk(denorm_noise,reference_data,output_noise_image_path)
    print(f"保存加噪后的图像到: {output_noise_image_path}")

def add_noise_to_image(orig_image_path, noise_image_path, output_noise_image_path, eps):
    # orig_image, reference_data = load_nifti_sitk(orig_image_path)
    # orig_image_noise,_ = load_nifti_sitk(orig_image_path.replace("nii_complete","tmp").replace("image","noise"))
    noise_image_noise,_ = load_nifti_sitk(noise_image_path.replace("nii_incomplete","tmp").replace("image","noise"))
    orig_denoise_image,_ = load_nifti_sitk(orig_image_path.replace("nii_complete","tmp").replace("image","denoised_image"))
    # noise_denoise_image,_ = load_nifti_sitk(noise_image_path.replace("nii_incomplete","noise_template").replace("image","denoised_image"))

    # try_noise = try4(noise_denoise_image,orig_denoise_image,noise_image_noise,orig_image_noise)
    try_noise = noise_image_noise

    add_noise_image = orig_denoise_image + eps*try_noise
    save_nifti_sitk(add_noise_image,None,output_noise_image_path)
    print(f"保存加噪后的图像到: {output_noise_image_path}")


eps = 0.1
orig_ID = "3-28-R"
noise_ID = "1.3.6.1.4.1.30071.8.181067518145894.662461547040.1"
path = "/media/HDD/fanlinqian/ossicular/data1_process/"
orig_image_path = path + f"nii_complete/{orig_ID}_image.nii.gz"
noise_image_path = path + f"nii_incomplete/{noise_ID}_image.nii.gz"
output_noise_image_path = path + f"tmp/{orig_ID}_{eps}noise_image.nii.gz"

add_noise_to_image(orig_image_path, noise_image_path, output_noise_image_path, eps)
