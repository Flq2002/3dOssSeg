import SimpleITK as sitk
import numpy as np
import cv2

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

# 图像归一化到 0-255 并转换为 uint8
def normalize_to_uint8(image_data):
    original_min = np.min(image_data)
    original_max = np.max(image_data)
    normalized = ((image_data - original_min) / (original_max - original_min) * 255).astype(np.uint8)
    return normalized, original_min, original_max

# 反归一化到原始范围
def denormalize_from_uint8(normalized_data, original_min, original_max):
    return (normalized_data.astype(np.float32) / 255) * (original_max - original_min) + original_min

# 使用 OpenCV 对切片进行 NLM 去噪
def denoise_slices(image_data, h=10, template_window_size=7, search_window_size=21):
    denoised_data = np.zeros_like(image_data, dtype=np.uint8)
    for i in range(image_data.shape[0]):  # 遍历每个切片
        slice_data = image_data[i, :, :]  # 提取单个切片
        # 使用 OpenCV 的 NLM 去噪
        denoised_slice = cv2.fastNlMeansDenoising(
            src=slice_data,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        denoised_data[i, :, :] = denoised_slice
    return denoised_data

# 对三个方向进行去噪
def denoise_3d(image_data, h=10, template_window_size=7, search_window_size=21):
    # 轴向方向（Axial: [Z, Y, X]）
    axial_denoised = denoise_slices(image_data, h, template_window_size, search_window_size)
    # 矢状方向（Sagittal: [X, Z, Y]）
    sagittal_data = np.transpose(image_data, (2, 0, 1))  # 重排轴顺序
    sagittal_denoised = denoise_slices(sagittal_data, h, template_window_size, search_window_size)
    sagittal_denoised = np.transpose(sagittal_denoised, (1, 2, 0))  # 恢复轴顺序

    # 冠状方向（Coronal: [Y, Z, X]）
    coronal_data = np.transpose(image_data, (1, 0, 2))  # 重排轴顺序
    coronal_denoised = denoise_slices(coronal_data, h, template_window_size, search_window_size)
    coronal_denoised = np.transpose(coronal_denoised, (1, 0, 2))  # 恢复轴顺序

    # 平均三方向的去噪结果
    denoised_combined = (axial_denoised.astype(np.float32) +
                         sagittal_denoised.astype(np.float32) +
                         coronal_denoised.astype(np.float32)) / 3
    return denoised_combined.astype(np.uint8)

def nlm_denoise(image_data):
    normalized_data, original_min, original_max = normalize_to_uint8(image_data)
    denoised_normalized = denoise_3d(
        normalized_data, h=15, template_window_size=7, search_window_size=21
    )
    denoised = denormalize_from_uint8(denoised_normalized, original_min, original_max)
    return denoised
# 主函数
from denoisy_bi import process_volume_all_directions
def process_nifti(input_path, denoised_output_path, noise_output_path, h=10, template_window_size=7, search_window_size=21):
    # 加载图像
    image_data, reference_image = load_nifti_sitk(input_path)

    # 归一化到 0-255 并转换为 uint8
    normalized_data, original_min, original_max = normalize_to_uint8(image_data)
    
    # 对三个方向进行去噪
    denoised_normalized = denoise_3d(
        normalized_data, h=h, template_window_size=template_window_size, search_window_size=search_window_size
    )

    # 计算噪声部分（原始 - 去噪）
    noise_normalized = normalized_data-denoised_normalized
    # from IPython import embed;embed()
    # 反归一化数据
    denoised = denormalize_from_uint8(denoised_normalized, original_min, original_max)
    # noise = denormalize_from_uint8(noise_normalized, original_min, original_max)
    noise = image_data - denoised

    # 保存去噪后图像
    save_nifti_sitk(denoised, reference_image, denoised_output_path)
    # print(f"Denoised image saved to: {denoised_output_path}")

    # 保存噪声图像
    save_nifti_sitk(noise, reference_image, noise_output_path)

    bi_noise = process_volume_all_directions(noise)
    save_nifti_sitk(bi_noise, reference_image, noise_output_path.replace("noise.nii.gz","bi_noise.nii.gz"))

    # save_nifti_sitk(image_data+noise, reference_image, noise_output_path.replace("noise","image"))
    # print(f"Noise image saved to: {noise_output_path}")

def read_list():
    import os
    ids_list = np.loadtxt(
        os.path.join('/media/HDD/fanlinqian/ossicular/data1_process/split_txts/complete_DG.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)
if __name__ == '__main__':
    
    # import os
    # from tqdm import tqdm
    # for filename in tqdm(os.listdir(path+"nii_complete")):
    #     if "image" in filename and filename.endswith(".nii.gz"):
    #         id = filename.split("_")[0]
    #         input_file = path+f"nii_complete/{id}_image.nii.gz"
    #         denoised_file = path+f"nii_complete/{id}_denoised_image.nii.gz"
    #         noise_file = path+f"nii_complete/{id}_noise.nii.gz"
    #         process_nifti(
    #             input_file,
    #             denoised_file,
    #             noise_file,
    #             h=15,  # 调整去噪强度
    #             template_window_size=7,
    #             search_window_size=21
    #         )
    # 输入的图像是没有经过归一化的
    # id = "1.3.6.1.4.1.30071.8.181067518145894.662461547040.1"
    # id = "3-28-R"
    ids = read_list()
    path = "/media/HDD/fanlinqian/ossicular/data1_process/"
    # input_file = path+f"nii_incomplete/{id}_image.nii.gz"  # 替换为实际输入文件路径
    # denoised_file = path+f"noise_template/{id}_denoised_image.nii.gz"
    # noise_file = path+f"noise_template/{id}_noise.nii.gz"
    # input_file = path+f"tmp/{id}_2noise_image.nii.gz"  # 替换为实际输入文件路径
    # input_file = "/media/HDD/fanlinqian/ossicular/data1_process/nii/2-1-L_image.nii.gz"
    from tqdm import tqdm
    for id in tqdm(ids):
        input_file = f"/media/HDD/fanlinqian/work_dirs_othermodels/Exp_OSS/20250218-1739846623-swin_unetr/predictions/{id}_image.nii.gz"
        denoised_file = path+f"noise_image/{id}_denoised_image.nii.gz"
        noise_file = path+f"noise_image/{id}_noise.nii.gz"
        process_nifti(
            input_file,
            denoised_file,
            noise_file,
            h=15,  # 调整去噪强度
            template_window_size=7,
            search_window_size=21
        )