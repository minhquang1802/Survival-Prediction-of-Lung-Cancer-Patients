import os

seg_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/Segmentation"

for file in os.listdir(seg_path):
    if file.endswith("_fixed.nii.gz"):
        original_file = file.replace("_fixed.nii.gz", ".nii.gz")
        original_path = os.path.join(seg_path, original_file)
        fixed_path = os.path.join(seg_path, file)

        if os.path.exists(original_path):
            print(f"Delete: {original_path}")
            os.remove(original_path)
            
        new_name = original_path
        os.rename(fixed_path, new_name)
        print(f" Rename {file} ➝ {original_file}")
