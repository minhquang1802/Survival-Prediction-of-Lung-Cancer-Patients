import numpy as np
import os

def check_empty_slices_in_folder(folder_path):
    """
    Check slices in all .npy files in a given folder to find empty slices.

    Args:
        folder_path (str): Path to the folder containing .npy files

    Returns:
        results (dict): Dictionary containing information about the checked files.
    """
    results = {}

    # Iterate through all .npy files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            mask_slices = np.load(file_path)

            # Check for empty slices
            empty_slices = [i for i in range(mask_slices.shape[0]) if np.all(mask_slices[i] == 0)]
            total_empty = len(empty_slices)
            total_slices = mask_slices.shape[0]

            # Save results to dictionary
            results[file_name] = {
                "total_slices": total_slices,
                "total_empty_slices": total_empty,
                "empty_slice_indices": empty_slices
            }

            # Print results for each file
            print(f"File: {file_name}")
            print(f"   - Shape of {file_path}: {mask_slices.shape}")
            print(f"   - Total slices: {total_slices}")
            print(f"   - Empty slices: {total_empty} ({(total_empty/total_slices)*100:.2f}%)")
            print(f"   - Indices of empty slices: {empty_slices if empty_slices else 'None'}\n")

    return results

# Run checks for all .npy files in the folder
folder_path = "../../..//FeatureExtraction/method_2/test"  # Replace with the path to the folder containing .npy files
check_empty_slices_in_folder(folder_path)
