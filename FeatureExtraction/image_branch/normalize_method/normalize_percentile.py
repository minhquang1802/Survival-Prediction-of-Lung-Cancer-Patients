import os
import numpy as np
from tqdm import tqdm  # Display progress bar

def compute_dataset_statistics(folder):
    """
    Calculate mean, std, and percentiles (0.5% and 99.5%) of the entire CT dataset
    """
    all_ct_data = []
    
    # Collect CT data from all files
    for filename in tqdm(os.listdir(folder), desc="Collecting Dataset Statistics"):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            data = np.load(file_path)
            
            # Only take CT channel (channel 0)
            ct_channel = data[:, :, :, 0].flatten()
            all_ct_data.append(ct_channel)
    
    # Combine all CT values into one large array
    all_ct_data = np.concatenate(all_ct_data, axis=0)
    
    # Calculate important statistics
    dataset_mean = np.mean(all_ct_data)
    dataset_std = np.std(all_ct_data)
    lower_bound = np.percentile(all_ct_data, 0.5)  # 0.5 percentile
    upper_bound = np.percentile(all_ct_data, 99.5) # 99.5 percentile
    
    print(f"Dataset Mean: {dataset_mean}")
    print(f"Dataset Std: {dataset_std}")
    print(f"0.5 Percentile: {lower_bound}")
    print(f"99.5 Percentile: {upper_bound}")
    
    return dataset_mean, dataset_std, lower_bound, upper_bound

def normalize_ct_nnunet(data, dataset_mean, dataset_std, lower_bound, upper_bound):
    """
    Normalize CT images according to nnUNet method
    """
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    # Get CT channel (channel 0)
    ct_data = data[:, :, :, 0]
    
    # Clip values according to percentile
    ct_data = np.clip(ct_data, lower_bound, upper_bound)
    
    # Normalize according to mean and std of entire dataset
    normalized_data[:, :, :, 0] = (ct_data - dataset_mean) / max(dataset_std, 1e-8)
    
    # Keep segmentation mask and distance transform unchanged
    normalized_data[:, :, :, 1] = data[:, :, :, 1]
    normalized_data[:, :, :, 2] = data[:, :, :, 2]
    
    return normalized_data

def process_npy_files_nnunet(folder):
    """
    Process normalize entire dataset according to nnUNet method
    """
    # Calculate statistics of entire dataset
    dataset_mean, dataset_std, lower_bound, upper_bound = compute_dataset_statistics(folder)
    
    # Normalize each file
    for filename in tqdm(os.listdir(folder), desc="Normalizing Files"):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            
            # Load file
            data = np.load(file_path)
            
            # Normalize
            normalized_data = normalize_ct_nnunet(data, dataset_mean, dataset_std, lower_bound, upper_bound)
            
            # Overwrite file
            np.save(file_path, normalized_data)
            
folder = "../../../FeatureExtraction/image_branch/method_2/percentile/data_percentile"
process_npy_files_nnunet(folder)
