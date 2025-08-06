import os
import nibabel as nib
import numpy as np

# Đường dẫn thư mục chứa file CT
ct_folder = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/Segmentation"

# Đường dẫn file log
log_file_path = "E:/DATN_LVTN/nnUNet/log/z_order_check.txt"

# Mở file log để ghi kết quả
with open(log_file_path, "w") as log_file:
    log_file.write("Z-Axis Order Check Log\n")
    log_file.write("=" * 50 + "\n")

    # Duyệt qua từng file trong thư mục CT
    for ct_file in os.listdir(ct_folder):
        if ct_file.endswith(".nii.gz"):
            ct_file_path = os.path.join(ct_folder, ct_file)
            
            try:
                # Load file NIfTI
                ct_image = nib.load(ct_file_path)
                affine = ct_image.affine
                
                # Lấy danh sách tọa độ Z của từng lát cắt
                num_slices = ct_image.shape[2]
                z_values = [affine[2, 3] + i * affine[2, 2] for i in range(num_slices)]

                # Kiểm tra thứ tự của giá trị Z
                if all(z_values[i] > z_values[i + 1] for i in range(len(z_values) - 1)):
                    log_file.write(f"{ct_file}: Z-axis is decreasing (expected order)\n")
                elif all(z_values[i] < z_values[i + 1] for i in range(len(z_values) - 1)):
                    log_file.write(f"{ct_file}: Z-axis is increasing (unexpected order)\n")
                else:
                    log_file.write(f"{ct_file}: Z-axis order is inconsistent\n")

            except Exception as e:
                log_file.write(f"{ct_file}: Error reading file - {str(e)}\n")

    log_file.write("=" * 50 + "\n")
    log_file.write("Check completed.\n")

print(f"Kiểm tra hoàn tất. Xem kết quả trong file log: {log_file_path}")
