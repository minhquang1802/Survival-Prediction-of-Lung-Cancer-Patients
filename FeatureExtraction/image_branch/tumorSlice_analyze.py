import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def count_slices_with_tumor(mask_path):
    """
    Count the number of slices in segmentation mask file that contain tumor.
    """
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()

    # Z-axis is axis=2 (slices in depth direction)
    num_slices_with_tumor = np.sum(np.any(mask_data > 0, axis=(0, 1)))

    return num_slices_with_tumor

def analyze_segmentation_masks(mask_dir):
    """
    Analyze the number of tumor-containing slices for all .nii.gz files in the directory.
    Draw boxplot and print Q1, Q2, Q3 along with the number of patients in each group.
    """
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")]

    slice_counts = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        num_slices = count_slices_with_tumor(mask_path)
        slice_counts.append(num_slices)
        print(f"{mask_file}: {num_slices} slices contain tumors")

    if not slice_counts:
        print("No segmentation files found.")
        return

    slice_counts = np.array(slice_counts)

    # Calculate Q1, Q2, Q3
    Q1 = np.percentile(slice_counts, 25)
    Q2 = np.median(slice_counts)
    Q3 = np.percentile(slice_counts, 75)

    # Count number of patients in each group
    count_Q1 = np.sum(slice_counts <= Q1)
    count_Q2 = np.sum((slice_counts > Q1) & (slice_counts <= Q2))
    count_Q3 = np.sum((slice_counts > Q2) & (slice_counts <= Q3))
    count_Q4 = np.sum(slice_counts > Q3)

    print("\n=============================")
    print(f"Q1 (25%): {Q1:.2f} slices")
    print(f"Median (Q2): {Q2:.2f} slices")
    print(f"Q3 (75%): {Q3:.2f} slices")
    print("Number of patients in each group:")
    print(f"  - Group 1 (≤ Q1): {count_Q1}")
    print(f"  - Group 2 (Q1 - Q2): {count_Q2}")
    print(f"  - Group 3 (Q2 - Q3): {count_Q3}")
    print(f"  - Group 4 (> Q3): {count_Q4}")
    print("=============================\n")

    # Draw boxplot with scatter
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=slice_counts, color='lightblue', width=0.3)
    sns.stripplot(y=slice_counts, color='darkblue', jitter=True, size=6, alpha=0.6)

    # Annotate Q1, Q2, Q3 markers
    plt.axhline(Q1, color='orange', linestyle='--', label=f'Q1 = {Q1:.2f}')
    plt.axhline(Q2, color='green', linestyle='--', label=f'Median = {Q2:.2f}')
    plt.axhline(Q3, color='red', linestyle='--', label=f'Q3 = {Q3:.2f}')
    plt.legend()

    plt.title("Distribution of tumor-containing slices per patient")
    plt.ylabel("Number of slices")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Path to directory containing segmentation masks
mask_dir = "../../FeatureExtraction/image_branch/dataset/segmentation_mask"
analyze_segmentation_masks(mask_dir)
