import os
import nibabel as nib
import numpy as np
from glob import glob

def flip_and_save_nii(input_folder, output_folder, flip_axes=(1, 2)):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Tìm tất cả file .nii.gz
    nii_files = sorted(glob(os.path.join(input_folder, "*.nii.gz")))

    if len(nii_files) == 0:
        print("Không tìm thấy file .nii.gz nào.")
        return

    print(f"Đã tìm thấy {len(nii_files)} file để xử lý...")

    for idx, file_path in enumerate(nii_files):
        try:
            print(f"[{idx+1}/{len(nii_files)}] Xử lý: {os.path.basename(file_path)}")

            # Load ảnh
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header

            # Lật ảnh theo từng chiều
            for axis in flip_axes:
                data = np.flip(data, axis=axis)

            # Tạo ảnh mới và lưu lại
            flipped_img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
            out_path = os.path.join(output_folder, os.path.basename(file_path))
            nib.save(flipped_img, out_path)

            print(f"Đã lưu ảnh đã lật tại: {out_path}")

        except Exception as e:
            print(f"Lỗi khi xử lý {file_path}: {str(e)}")

    print("Hoàn thành lật toàn bộ ảnh.")

# === Cách sử dụng ===
if __name__ == "__main__":
    input_dir = "E:/DATN_LVTN/TumorSegmentation/AttenUNet_2/inference"             # thư mục chứa các ảnh .nii.gz gốc
    output_dir = "E:/DATN_LVTN/TumorSegmentation/AttenUNet_2/inference_flipped"    # thư mục để lưu ảnh đã lật
    flip_and_save_nii(input_dir, output_dir, flip_axes=(0, 1))  # lật theo trục Y, rồi X
