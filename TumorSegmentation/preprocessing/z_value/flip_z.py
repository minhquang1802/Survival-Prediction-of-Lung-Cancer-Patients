# đảo thứ tự giá trị z trong các slice (đảo slice)
import nibabel as nib
import numpy as np
import os

# Đường dẫn đến file NIfTI gốc
nifti_file_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/Segmentation/lung_023.nii.gz"

# Load file NIfTI
ct_image = nib.load(nifti_file_path)
data = ct_image.get_fdata()  # Dữ liệu ảnh
affine = ct_image.affine  # Ma trận affine

# Đảo ngược trục Z (axis=2)
flipped_data = np.flip(data, axis=2)

# Cập nhật ma trận affine để phản ánh sự thay đổi
new_affine = affine.copy()
new_affine[2, 2] *= -1  # Đảo dấu giá trị spacing theo Z
new_affine[2, 3] = affine[2, 3] + (data.shape[2] - 1) * affine[2, 2]  # Cập nhật origin

# Tạo file NIfTI mới với lát cắt đảo ngược
flipped_nifti = nib.Nifti1Image(flipped_data, new_affine, ct_image.header)

# Lưu file mới
output_path = nifti_file_path.replace(".nii.gz", "_flipped.nii.gz")
nib.save(flipped_nifti, output_path)

print(f" File mới đã được lưu: {output_path}")
