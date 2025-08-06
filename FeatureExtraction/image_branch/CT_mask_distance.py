import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

def select_slices(mask_3d):
    """
    Select slices containing tumors based on predefined criteria.
    """
    tumor_slices = [i for i in range(mask_3d.shape[2]) if np.any(mask_3d[:, :, i])]
    num_tumor_slices = len(tumor_slices)
    
    if num_tumor_slices == 0:
        return np.array([]), []
    
    if num_tumor_slices <= 21:
        selected_indices = tumor_slices
    elif 22 <= num_tumor_slices <= 36:
        selected_indices = select_around_max(mask_3d, tumor_slices, center_slices=25)
    else:
        selected_indices = select_around_max(mask_3d, tumor_slices, center_slices=35)
    
    return mask_3d[:, :, selected_indices], selected_indices

def select_around_max(mask_3d, tumor_slices, center_slices=15):
    """
    Select a fixed number of slices around the slice with the largest tumor region.
    """
    max_slice_idx = max(tumor_slices, key=lambda i: np.sum(mask_3d[:, :, i] > 0))
    center_slices = min(center_slices, len(tumor_slices))
    tumor_slices = sorted(tumor_slices)
    
    half = center_slices // 2
    available_before = max_slice_idx - max(0, tumor_slices[0])
    available_after = min(mask_3d.shape[2] - 1, tumor_slices[-1]) - max_slice_idx

    if available_before >= half and available_after >= half:
        start = max_slice_idx - half
        end = max_slice_idx + half + 1
    elif available_before < half:
        start = max_slice_idx - available_before
        end = start + center_slices
    else:
        end = max_slice_idx + available_after + 1
        start = end - center_slices

    return list(range(start, end))

def create_distance_transform(binary_mask):
    """
    Create a distance transform from a binary mask, normalized by the diagonal of the image.
    """
    binary = binary_mask > 0
    dist_transform = distance_transform_edt(binary)

    h, w = binary_mask.shape
    diagonal = np.sqrt(h**2 + w**2)
    normalized_dist = dist_transform / diagonal
    
    return normalized_dist

def process_and_save_slices(ct_dir, mask_dir, output_dir):
    """
    Process CT and segmentation mask files, select slices, create data, and save as .npy.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(mask_dir):
        if filename.endswith(".nii.gz"):
            mask_path = os.path.join(mask_dir, filename)
            patient_id = filename.replace(".nii.gz", "")  
            ct_filename = f"{patient_id}_0000.nii.gz"
            ct_path = os.path.join(ct_dir, ct_filename)
            output_path = os.path.join(output_dir, filename.replace(".nii.gz", ".npy"))
            
            if not os.path.exists(ct_path):
                print(f"Missing corresponding CT file for {filename}")
                continue
            
            print(f"Processing {filename}...")
            
            mask_nii = nib.load(mask_path)
            mask_3d = mask_nii.get_fdata()
            
            ct_nii = nib.load(ct_path)
            ct_3d = ct_nii.get_fdata()
            
            selected_slices, selected_indices = select_slices(mask_3d)
            
            if selected_slices.size == 0:
                print(f"No tumor slices found in {filename}")
                continue
            
            # Extract corresponding CT slices
            ct_slices = ct_3d[:, :, selected_indices]
            
            # Generate distance transform
            distance_maps = np.array([create_distance_transform(mask_3d[:, :, i]) for i in selected_indices])
            distance_maps = distance_maps.transpose(1, 2, 0)
            
            # Combine into (num_slices, H, W, 3)
            combined_slices = np.stack([ct_slices, selected_slices, distance_maps], axis=-1)
            
            # Save as .npy
            np.save(output_path, combined_slices)
            print(f"Saved {output_path} with shape {combined_slices.shape}")


ct_dir = "../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/imagesTs"
mask_dir = "../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTs"
output_dir = "../../FeatureExtraction/image_branch/method_2/percentile/data_percentile"

process_and_save_slices(ct_dir, mask_dir, output_dir)
