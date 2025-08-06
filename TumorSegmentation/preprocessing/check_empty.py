import os
import nibabel as nib
import numpy as np

# Đường dẫn tới folder Segmentation
segmentation_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/Segmentation"

# Danh sách lưu các file segmentation rỗng
empty_masks = []

# Duyệt qua từng file trong folder Segmentation
for seg_file in os.listdir(segmentation_path):
    if seg_file.endswith(".nii.gz"):
        seg_file_path = os.path.join(segmentation_path, seg_file)
        
        # Đọc file segmentation
        seg_image = nib.load(seg_file_path)
        seg_data = seg_image.get_fdata()
        
        # Kiểm tra nếu segmentation mask chỉ chứa toàn giá trị 0
        if np.all(seg_data == 0):
            print(f"Empty segmentation mask: {seg_file}")
            empty_masks.append(seg_file)

# In kết quả
if empty_masks:
    print("\n Found empty segmentation masks:")
    for mask in empty_masks:
        print(f"- {mask}")
else:
    print("\n All segmentation masks contain tumor data.")
