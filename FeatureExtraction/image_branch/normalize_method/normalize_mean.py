import os
import numpy as np
from tqdm import tqdm

def compute_dataset_statistics(folder):
    """
    Calculate mean and std of the entire CT dataset
    
    Parameters:
    -----------
    folder : str
        Path to folder containing .npy files
    
    Returns:
    --------
    tuple: (dataset_mean, dataset_std)
    """
    all_ct_data = []
    
    # Collect CT data from all files
    for filename in tqdm(os.listdir(folder), desc="Collecting Dataset Statistics"):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            data = np.load(file_path)
            
            # Only take CT channel (channel 0)
            ct_channel = data[:, :, :, 0]
            all_ct_data.append(ct_channel)
    
    # Stack all data
    all_ct_data = np.concatenate(all_ct_data, axis=2)
    
    # Calculate mean and std on entire dataset
    dataset_mean = np.mean(all_ct_data)
    dataset_std = np.std(all_ct_data)
    
    print(f"Dataset Mean: {dataset_mean}")
    print(f"Dataset Std: {dataset_std}")
    
    return dataset_mean, dataset_std

def normalize_ct_dataset_wise(data, dataset_mean, dataset_std):
    """
    Normalize CT images according to entire dataset statistics
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data tensor with shape [512, 512, num_slices, 3]
    dataset_mean : float
        Mean value of entire dataset
    dataset_std : float
        Standard deviation of entire dataset
    
    Returns:
    --------
    numpy.ndarray
        Normalized data tensor
    """
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    # Normalize CT scans (channel 0) 
    normalized_data[:, :, :, 0] = (data[:, :, :, 0] - dataset_mean) / dataset_std
    
    # Keep segmentation mask and distance transform unchanged
    normalized_data[:, :, :, 1] = data[:, :, :, 1]
    normalized_data[:, :, :, 2] = data[:, :, :, 2]
    
    return normalized_data

def process_npy_files_dataset_wise(folder):
    """
    Process normalize entire dataset
    """
    # Calculate statistics of entire dataset
    dataset_mean, dataset_std = compute_dataset_statistics(folder)
    
    # Normalize each file
    for filename in tqdm(os.listdir(folder), desc="Normalizing Files"):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            
            # Load file
            data = np.load(file_path)
            
            # Normalize
            normalized_data = normalize_ct_dataset_wise(data, dataset_mean, dataset_std)
            
            # Overwrite file
            np.save(file_path, normalized_data)
            
folder = "../../../FeatureExtraction/image_branch/method_2/data"
process_npy_files_dataset_wise(folder)