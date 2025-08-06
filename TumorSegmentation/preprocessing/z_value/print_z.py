# in ra tất cả giá trị z trong 1 file nifti
import nibabel as nib

# Đường dẫn đến file NIfTI cần kiểm tra
nifti_file_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/Segmentation/lung_023_flipped.nii.gz"

# Load file NIfTI
ct_image = nib.load(nifti_file_path)
affine = ct_image.affine  # Ma trận affine
num_slices = ct_image.shape[2]  # Số lát cắt theo trục Z

# Lấy danh sách tọa độ Z của từng lát cắt
z_values = [affine[2, 3] + i * affine[2, 2] for i in range(num_slices)]

# In kết quả
print(f"Danh sách tọa độ Z của file: {nifti_file_path}")
for i, z in enumerate(z_values):
    print(f"Slice {i + 1}: Z = {z}")
