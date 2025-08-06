"""
Chọn lọc các lát cắt chứa khối u từ segmentation mask và lưu dưới dạng file .npy.
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import cv2

def select_slices(mask_3d):
    """
    Chọn lọc các lát cắt chứa khối u theo chiến lược phân nhóm đã nêu.
    
    Args:
        mask_3d (numpy array): Mảng 3D chứa segmentation mask (shape: H, W, D).

    Returns:
        selected_slices (numpy array): Các lát cắt đã chọn.
    """
    tumor_slices = [i for i in range(mask_3d.shape[2]) if np.any(mask_3d[:, :, i])]
    num_tumor_slices = len(tumor_slices)
    
    if num_tumor_slices == 0:
        return np.array([])
    
    if num_tumor_slices <= 21:
        selected_indices = tumor_slices
    elif 22 <= num_tumor_slices <= 36:
        selected_indices = select_around_max(mask_3d, tumor_slices, center_slices=25)
    else:
        selected_indices = select_around_max(mask_3d, tumor_slices, center_slices=35)
    

    selected_slices = mask_3d[:, :, selected_indices]
    return selected_slices, selected_indices

def select_around_max(mask_3d, tumor_slices, center_slices=15):
    """
    Select a fixed number of slices around the slice with the largest tumor region.
    
    Args:
        mask_3d (numpy array): 3D array of segmentation mask.
        tumor_slices (list): List of slice indices containing tumors.
        center_slices (int): Total number of slices to select.

    Returns:
        list: List of selected slice indices.
    """
    # Find the slice with the largest tumor area
    max_slice_idx = max(tumor_slices, key=lambda i: np.sum(mask_3d[:, :, i] > 0))
    center_slices = min(center_slices, len(tumor_slices))
    tumor_slices = sorted(tumor_slices)
    
    half = center_slices // 2
    available_before = max_slice_idx - max(0, tumor_slices[0])  # Slices available before
    available_after = min(mask_3d.shape[2] - 1, tumor_slices[-1]) - max_slice_idx  # Slices available after

    # Determine how many slices to take from each side
    if available_before >= half and available_after >= half:
        start = max_slice_idx - half
        end = max_slice_idx + half + 1
    elif available_before < half:  # Not enough slices before, take more from after
        start = max_slice_idx - available_before
        end = start + center_slices
    else:  # Not enough slices after, take more from before
        end = max_slice_idx + available_after + 1
        start = end - center_slices

    return list(range(start, end))

def create_distance_transform(binary_mask):
    """
    Tạo distance transform từ binary mask.
    
    Args:
        binary_mask (numpy array): Binary mask 2D.
        
    Returns:
        distance_map: Distance transform được chuẩn hóa.
    """
    # Đảm bảo mask là binary
    binary = binary_mask > 0
    
    # Tính distance transform (khoảng cách từ mỗi pixel đến pixel = 0 gần nhất)
    dist_transform = distance_transform_edt(binary)
    
    # Chuẩn hóa về khoảng [0, 1]
    if dist_transform.max() > 0:
        dist_transform = dist_transform / dist_transform.max()
        
    return dist_transform

def create_boundary_map(binary_mask, thickness=1):
    """
    Tạo boundary map từ binary mask.
    
    Args:
        binary_mask (numpy array): Binary mask 2D.
        thickness (int): Độ dày của boundary.
        
    Returns:
        boundary_map: Boundary map với các viền của vùng quan tâm.
    """
    # Đảm bảo mask là binary và đúng định dạng
    binary = (binary_mask > 0).astype(np.uint8)
    
    if np.all(binary == 0):
        print("Warning: No foreground pixels found in mask.")
        return np.zeros_like(binary, dtype=np.uint8)
    
    # Tìm các contour
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found in mask.")
        return np.zeros_like(binary, dtype=np.uint8)
    
    # Tạo ảnh trống để vẽ boundary
    boundary = np.zeros_like(binary, dtype=np.uint8)
    
    print("Mask shape:", binary.shape)
    print("Unique values in mask:", np.unique(binary))
    print("Contours found:", len(contours))
    print("Boundary dtype:", boundary.dtype)
    
    # Vẽ các contour lên ảnh
    cv2.drawContours(boundary, contours, -1, 1, thickness)
    
    return boundary

def create_multichannel_slices(selected_slices):
    """
    Tạo dữ liệu đa kênh từ các lát cắt đã chọn.
    
    Args:
        selected_slices (numpy array): Các lát cắt đã chọn (H, W, num_slices).
        
    Returns:
        multichannel_data: Dữ liệu đa kênh (num_slices, H, W, 3).
    """
    H, W, num_slices = selected_slices.shape
    multichannel_data = np.zeros((num_slices, H, W, 3))
    
    for i in range(num_slices):
        # Kênh 1: Mask gốc
        mask = selected_slices[:, :, i]
        multichannel_data[i, :, :, 0] = mask
        
        # Kênh 2: Distance transform
        multichannel_data[i, :, :, 1] = create_distance_transform(mask)
        
        # Kênh 3: Boundary map
        multichannel_data[i, :, :, 2] = create_boundary_map(mask)
    
    return multichannel_data


def process_and_save_multichannel(input_dir, output_dir):
    """
    Duyệt qua tất cả các file .nii.gz trong thư mục, thực hiện chọn lọc lát cắt,
    tạo dữ liệu đa kênh và lưu dưới dạng file .npy.

    Args:
        input_dir (str): Đường dẫn chứa các file segmentation .nii.gz.
        output_dir (str): Đường dẫn để lưu file .npy.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".nii.gz", "_multichannel.npy"))

            print(f"Đang xử lý {filename}...")
            
            # Load file
            nii_image = nib.load(file_path)
            mask_3d = nii_image.get_fdata()

            selected_slices,_ = select_slices(mask_3d)

            if selected_slices.size == 0:
                print(f"{filename} không có lát cắt chứa khối u")
                continue
            
            # Tạo dữ liệu đa kênh
            multichannel_data = create_multichannel_slices(selected_slices)

            # Lưu file .npy
            np.save(output_path, multichannel_data)
            print(f"Đã lưu {output_path} với shape {multichannel_data.shape}")

input_dir = "../../FeatureExtraction/segmentation_mask/train"
output_dir = "../../FeatureExtraction/raw_npy/train"


process_and_save_multichannel(input_dir, output_dir)