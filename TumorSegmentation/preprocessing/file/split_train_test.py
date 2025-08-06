import os
import shutil
import random

# Path
base_path = "E:/DATN_LVTN/nnUNet/train/nnUNet_raw/Dataset015_lungTumor"
ct_path = os.path.join(base_path, "CT")
seg_path = os.path.join(base_path, "Segmentation")

# Output Path
imagesTr_path = os.path.join(base_path, "imagesTr")
labelsTr_path = os.path.join(base_path, "labelsTr")
imagesTs_path = os.path.join(base_path, "imagesTs")
labelsTs_path = os.path.join(base_path, "labelsTs")

# Create folders
for folder in [imagesTr_path, labelsTr_path, imagesTs_path, labelsTs_path]:
    os.makedirs(folder, exist_ok=True)

# Get patients list
patients = [f.replace("lung_", "").replace("_0000.nii.gz", "") for f in os.listdir(ct_path) if f.endswith(".nii.gz")]

# 80 train - 20 test
random.seed(42)  # random
random.shuffle(patients)
train_patients = patients[:int(0.8 * len(patients))]  # 115 (80%)
test_patients = patients[int(0.8 * len(patients)):]   # 29 (20%)

def copy_files(patients_list, dest_ct, dest_seg):
    for patient_id in patients_list:
        ct_file = f"lung_{patient_id}_0000.nii.gz"
        seg_file = f"lung_{patient_id}.nii.gz"
        
        # Kiểm tra file tồn tại trước khi sao chép
        if os.path.exists(os.path.join(ct_path, ct_file)) and os.path.exists(os.path.join(seg_path, seg_file)):
            shutil.copy(os.path.join(ct_path, ct_file), os.path.join(dest_ct, ct_file))
            shutil.copy(os.path.join(seg_path, seg_file), os.path.join(dest_seg, seg_file))
            print(f"Copied {ct_file} & {seg_file}")

copy_files(train_patients, imagesTr_path, labelsTr_path)

copy_files(test_patients, imagesTs_path, labelsTs_path)

print("Dataset splitting completed")
