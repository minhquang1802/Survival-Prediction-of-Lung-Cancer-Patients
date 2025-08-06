import os
import SimpleITK as sitk

input_dir = "E:/DATN_LVTN/TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTs"
output_dir = input_dir

for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz"):
        file_path = os.path.join(input_dir, filename)
        
        # Đọc ảnh
        img = sitk.ReadImage(file_path)
        img_array = sitk.GetArrayFromImage(img)

        # Chuyển giá trị 255 thành 1
        img_array[img_array == 255] = 1

        new_img = sitk.GetImageFromArray(img_array)
        new_img.CopyInformation(img)
        sitk.WriteImage(new_img, os.path.join(output_dir, filename))

