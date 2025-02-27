import SimpleITK as sitk
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

# 加载 NIfTI 图像
def load_nifti_sitk(file_path):
    image = sitk.ReadImage(file_path)
    image_data = sitk.GetArrayFromImage(image)  # 转为 NumPy 数组
    return image_data, image

# 保存 NIfTI 图像
def save_nifti_sitk(data, reference_image, output_path):
    output_image = sitk.GetImageFromArray(data)  # 转为 SimpleITK 图像
    output_image.CopyInformation(reference_image)  # 复制空间信息
    sitk.WriteImage(output_image, output_path)

# 归一化图像
def normalize_image(image_data, min_val=0, max_val=1):
    original_min = np.min(image_data)
    original_max = np.max(image_data)
    normalized = (image_data - original_min) / (original_max - original_min) * (max_val - min_val) + min_val
    return normalized, original_min, original_max

# 恢复图像
def denormalize_image(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min

# 去噪处理
def denoise_image(image_data, patch_size=7, patch_distance=14, h_factor=0.8):
    # 估计噪声标准差（使用归一化数据）
    sigma_est = np.mean(estimate_sigma(image_data, multichannel=False))
    # 非局部均值去噪
    denoised = denoise_nl_means(
        image_data,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h_factor * sigma_est,
        multichannel=False
    )
    # 计算噪声部分
    noise = image_data - denoised
    return denoised, noise


# 主函数
def process_nifti(input_path, denoised_output_path, noise_output_path):
    # 加载图像
    image_data, reference_image = load_nifti_sitk(input_path)

    # 归一化处理
    normalized_data, original_min, original_max = normalize_image(image_data, 0, 255)

    # 去噪处理（在归一化范围内）
    denoised_normalized, noise_normalized = denoise_image(normalized_data)

    # 反归一化数据
    denoised = denormalize_image(denoised_normalized, original_min, original_max)
    # noise = denormalize_image(noise_normalized, original_min, original_max)

    # 保存去噪后图像
    save_nifti_sitk(denoised, reference_image, denoised_output_path)
    print(f"Denoised image saved to: {denoised_output_path}")

    # 保存噪声图像
    save_nifti_sitk(noise_normalized, reference_image, noise_output_path)
    print(f"Noise image saved to: {noise_output_path}")

# 使用示例
path = "/media/HDD/fanlinqian/ossicular/data1_process/"
input_file = path+"nii_complete/2-F5-R_image.nii.gz"  # 替换为实际输入文件路径
denoised_file = path+"tmp/2-F5-R_denoised_image.nii.gz"
noise_file = path+"tmp/2-F5-R_noise_image.nii.gz"
import time
start_time = time.time()
process_nifti(input_file, denoised_file, noise_file)
end_time = time.time()
print(f"耗时: {end_time - start_time:.6f} 秒")