import os
import numpy as np

def normalize_ct(ct_slice, window_min=-1200, window_max=600):
    # Clip values to standard lung window
    clipped = np.clip(ct_slice, window_min, window_max)
    # Normalize to [0, 1]
    normalized = (clipped - window_min) / (window_max - window_min)
    return normalized

def normalize_lung_data(data):
    """
    Normalize lung data with shape [512, 512, num_slices, 3]
    """
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    # Normalize CT scans (channel 0)
    for i in range(data.shape[2]):  # Iterate through each slice
        normalized_data[:, :, i, 0] = normalize_ct(
            data[:, :, i, 0], 
            window_min=-1200, 
            window_max=600
        )
    
    # Keep segmentation mask unchanged (channel 1)
    normalized_data[:, :, :, 1] = data[:, :, :, 1]
    
    # Keep distance transform unchanged (channel 2) as it's already in range [0, 1]
    normalized_data[:, :, :, 2] = data[:, :, :, 2]
    
    return normalized_data
        

def process_npy_files(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)

            # Load .npy file
            data = np.load(file_path)
            
            # data = process_background(data)
            normalized_data = normalize_lung_data(data)
            
            if normalized_data is not None:
                # Overwrite the file
                np.save(file_path, normalized_data)
                print(f"Processed and overwritten: {file_path}")
            
folder = "../../../FeatureExtraction/method_2/test"
process_npy_files(folder)
