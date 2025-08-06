import numpy as np

def save_npy_pixel_values(npy_path, output_txt_path):
    data = np.load(npy_path)  # Shape: (H, W, num_slices, 3)
    
    with open(output_txt_path, "w") as f:
        f.write(f"Loaded .npy file with shape: {data.shape}\n")

        num_slices = data.shape[2]  # number of slices
        for i in range(num_slices):
            f.write(f"\nSlice {i}:\n")
            f.write("CT channel:\n")
            np.savetxt(f, data[:, :, i, 0], fmt="%.4f")  # save CT values
            f.write("\nMask channel:\n")
            np.savetxt(f, data[:, :, i, 1], fmt="%.4f")  # save segmentation mask values
            f.write("\nDistance transform channel:\n")
            np.savetxt(f, data[:, :, i, 2], fmt="%.4f")  # save distance transform values
            f.write("\n" + "-"*80 + "\n")  # Separator between slices

# Call the function to check and save to file
save_npy_pixel_values("../../../FeatureExtraction/method_2/train/lung_002.npy", "npy_pixel_values.txt")
