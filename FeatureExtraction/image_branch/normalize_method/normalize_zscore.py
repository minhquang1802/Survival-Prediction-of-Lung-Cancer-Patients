import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_ct_zscore(ct_slice):
    """
    Perform Z-score normalization for CT slice
    
    Parameters:
    -----------
    ct_slice : numpy.ndarray
        Input CT image slice
    
    Returns:
    --------
    numpy.ndarray
        CT slice normalized using Z-score
    """
    # Flatten for calculation, then reshape back
    flat_slice = ct_slice.flatten()
    
    # Use StandardScaler to perform Z-score normalization
    scaler = StandardScaler()
    normalized_flat = scaler.fit_transform(flat_slice.reshape(-1, 1)).flatten()
    
    return normalized_flat.reshape(ct_slice.shape)

def normalize_lung_data_zscore(data):
    """
    Normalize lung data with shape [512, 512, num_slices, 3]
    Using Z-score normalization for CT scans
    """
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    # Normalize CT scans (channel 0) using Z-score
    for i in range(data.shape[2]):  # Iterate through each slice
        normalized_data[:, :, i, 0] = normalize_ct_zscore(data[:, :, i, 0])
    
    # Keep segmentation mask unchanged (channel 1)
    normalized_data[:, :, :, 1] = data[:, :, :, 1]
    
    # Keep distance transform unchanged (channel 2)
    normalized_data[:, :, :, 2] = data[:, :, :, 2]
    
    return normalized_data

def process_npy_files(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)

            # Load .npy file
            data = np.load(file_path)
            
            normalized_data = normalize_lung_data_zscore(data)
            
            if normalized_data is not None:
                # Overwrite the file
                np.save(file_path, normalized_data)
                print(f"Processed and overwritten: {file_path}")
            
folder = "../../../FeatureExtraction/image_branch/method_2/data"
process_npy_files(folder)