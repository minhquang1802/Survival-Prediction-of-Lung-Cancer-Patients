import nibabel as nib

# Đường dẫn tới file .nii.gz
# nifti_file = "//wsl.localhost/Ubuntu-22.04/home/quang/ML/nnU-Net/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/imagesTr/lung_080_0000.nii.gz"
nifti_file = "//wsl.localhost/Ubuntu-22.04/home/quang/ML/nnU-Net/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung/labelsTr/lung_080.nii.gz"

# Đọc file NIfTI
nifti_image = nib.load(nifti_file)

# Lấy thông tin về dimensions (shape)
shape = nifti_image.shape
print(f"Dimensions (X, Y, Z): {shape}")

# Lấy thông tin về pixel spacing và slice thickness (spacing)
spacing = nifti_image.header.get_zooms()
print(f"Spacing (Pixel Spacing, Slice Thickness): {spacing}")

# Lấy thông tin về affine matrix (origin và orientation)
affine = nifti_image.affine

# Lấy Origin (tọa độ gốc) từ affine matrix
origin = affine[:3, 3]

# In ra Origin
print(f"Affine Matrix:\n{affine}")
print(f"Origin: {origin}")

# Lấy thông tin về kiểu dữ liệu (data type)
data_type = nifti_image.get_data_dtype()
print(f"Data Type: {data_type}")

# Lấy giá trị cường độ (intensity values) tại các voxel
data = nifti_image.get_fdata()
print(f"Intensity Value at [0, 0, 0]: {data[0, 0, 0]}")
