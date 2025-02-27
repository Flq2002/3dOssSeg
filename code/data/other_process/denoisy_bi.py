import SimpleITK as sitk
import numpy as np
import cv2


# 读取 nii.gz 文件
def read_nii(filepath):
    image = sitk.ReadImage(filepath)
    image_array = sitk.GetArrayFromImage(image)  # (depth, height, width)
    return image, image_array


# 使用 OpenCV 的双边滤波对单个切片进行处理
def apply_bilateral_filter(slice_data, d=5, sigma_color=21, sigma_space=21):
    filtered_slice = cv2.bilateralFilter(slice_data, d, sigma_color, sigma_space)
    return filtered_slice

# 图像归一化到 0-255 并转换为 uint8
def normalize_to_uint8(image_data):
    original_min = np.min(image_data)
    original_max = np.max(image_data)
    normalized = ((image_data - original_min) / (original_max - original_min) * 255).astype(np.uint8)
    return normalized, original_min, original_max

# 反归一化到原始范围
def denormalize_from_uint8(normalized_data, original_min, original_max):
    return (normalized_data.astype(np.float32) / 255) * (original_max - original_min) + original_min

# 对某个方向的切片进行处理
def process_directional(image_array, axis):
    processed_slices = []
    slices = np.moveaxis(image_array, axis, 0)  # 将指定轴移动到第一维
    for i in range(slices.shape[0]):
        slice_data = slices[i, :, :]
        # 将切片数据归一化为 uint8
        slice_data_uint8, original_min, original_max = normalize_to_uint8(slice_data)
        filtered_slice = apply_bilateral_filter(slice_data_uint8)
        # 恢复原始范围
        filtered_slice_rescaled = denormalize_from_uint8(filtered_slice, original_min, original_max)
        processed_slices.append(filtered_slice_rescaled)
    return np.moveaxis(np.array(processed_slices), 0, axis)  # 将切片还原到原位置


# 对三个方向（轴向、矢状面、冠状面）分别进行处理
def process_volume_all_directions(image_array):
    processed_axial = process_directional(image_array, axis=0)  # Axial (depth方向)
    processed_sagittal = process_directional(image_array, axis=1)  # Sagittal (height方向)
    processed_coronal = process_directional(image_array, axis=2)  # Coronal (width方向)

    # 合并结果（取平均值以融合三方向信息）
    processed_combined = (processed_axial + processed_sagittal + processed_coronal) / 3
    return processed_combined


# 保存为新的 nii.gz 文件
def save_nii(filepath, reference_image, processed_array):
    processed_image = sitk.GetImageFromArray(processed_array)  # 转回 SimpleITK 图像
    processed_image.CopyInformation(reference_image)  # 保留原图的元信息
    sitk.WriteImage(processed_image, filepath)


# 主函数
def main(input_path, output_path):
    # 读取输入 nii.gz 文件
    reference_image, image_array = read_nii(input_path)

    # 应用三方向双边滤波
    processed_array = process_volume_all_directions(image_array)

    # 保存处理后的数据为新的 nii.gz 文件
    save_nii(output_path, reference_image, processed_array)
    print(f"处理完成，文件已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    id = "2-2-L"
    path = "/media/HDD/fanlinqian/ossicular/data1_process/noise_template/"
    input_file = path+f"{id}_noise.nii.gz"  # 输入文件路径
    output_file = path+f"{id}_noise_filter.nii.gz"  # 输出文件路径
    main(input_file, output_file)
