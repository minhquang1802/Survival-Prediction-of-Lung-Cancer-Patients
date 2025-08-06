import numpy as np
import os

# Folder path containing .npy files
folder_path = "../../../FeatureExtraction/method_2/train"

# Initialize min and max values for CT image (channel 0) and Distance Transform (channel 3)
global_min_ct = float("inf")
global_max_ct = float("-inf")
global_min_dt = float("inf")
global_max_dt = float("-inf")

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".npy"):
        file_path = os.path.join(folder_path, file_name)
        
        # Load file npy
        data = np.load(file_path)
        
        # Get channel 0 (CT image) and channel 3 (Distance Transform)
        ct_image = data[..., 0]
        distance_transform = data[..., 2]

        # Find min, max in current image
        file_min_ct = np.min(ct_image)
        file_max_ct = np.max(ct_image)
        file_min_dt = np.min(distance_transform)
        file_max_dt = np.max(distance_transform)
        
        # Update global min and max values
        global_min_ct = min(global_min_ct, file_min_ct)
        global_max_ct = max(global_max_ct, file_max_ct)
        global_min_dt = min(global_min_dt, file_min_dt)
        global_max_dt = max(global_max_dt, file_max_dt)

        print(f"File: {file_name} - CT Min: {file_min_ct}, CT Max: {file_max_ct}, DT Min: {file_min_dt}, DT Max: {file_max_dt}")

print("\n=== GLOBAL STATISTICS ===")
print(f"Global CT Min: {global_min_ct}")
print(f"Global CT Max: {global_max_ct}")
print(f"Global DT Min: {global_min_dt}")
print(f"Global DT Max: {global_max_dt}")
