import os
import nibabel as nib
import numpy as np

# Đường dẫn tới folder gốc
base_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor"
ct_path = os.path.join(base_path, "CT")
seg_path = os.path.join(base_path, "Segmentation")
log_file = "E:/DATN_LVTN/nnUNet/log/mismatch_slice.txt"

def check_and_log_mismatches(ct_path, seg_path, log_file):
    # Mở file log để ghi lại thông tin mismatch
    with open(log_file, "w") as log:
        log.write("Patient ID\tIssue\tDetails\n")

        for ct_file in os.listdir(ct_path):
            if ct_file.endswith(".nii.gz"):
                # Lấy ID bệnh nhân từ tên file
                patient_id = ct_file.replace("lung_", "").replace("_0000.nii.gz", "")

                # Tạo đường dẫn tới file CT và Segmentation tương ứng
                ct_file_path = os.path.join(ct_path, ct_file)
                seg_file_path = os.path.join(seg_path, f"lung_{patient_id}.nii.gz")

                # Kiểm tra nếu file Segmentation tồn tại
                if os.path.exists(seg_file_path):
                    # Đọc file CT và Segmentation
                    ct_image = nib.load(ct_file_path)
                    seg_image = nib.load(seg_file_path)

                    # Lấy thông tin Dimension, Spacing, Slice Thickness và Origin
                    ct_dimension = ct_image.shape
                    seg_dimension = seg_image.shape

                    ct_spacing = ct_image.header.get_zooms()
                    seg_spacing = seg_image.header.get_zooms()

                    ct_origin = ct_image.affine[:3, 3]
                    seg_origin = seg_image.affine[:3, 3]

                    # Kiểm tra Spacing
                    if not np.allclose(ct_spacing, seg_spacing):
                        log.write(f"{patient_id}\tSpacing Mismatch\tCT={ct_spacing}, Segmentation={seg_spacing}\n")

                    # Kiểm tra Origin
                    if not np.allclose(ct_origin, seg_origin):
                        log.write(f"{patient_id}\tOrigin Mismatch\tCT={ct_origin}, Segmentation={seg_origin}\n")

                    # Kiểm tra Dimension
                    if ct_dimension != seg_dimension:
                        log.write(f"{patient_id}\tDimension Mismatch\tCT={ct_dimension}, Segmentation={seg_dimension}\n")
    print(f"✅ Mismatch log saved to {log_file}")


check_and_log_mismatches(ct_path, seg_path, log_file)