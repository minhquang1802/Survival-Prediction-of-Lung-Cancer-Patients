import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np

# Đường dẫn tới folder chứa DICOM và NIfTI
dicom_base_path = "E:/DATN_LVTN/dataset/NSCLC_Radiogenomics/manifest-1732266091299/NSCLC Radiogenomics"
nifti_ct_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/CT"
nifti_seg_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor/Segmentation"

# Hàm để đọc thông số từ folder DICOM
def get_dicom_ct_info(dicom_folder):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    dimensions = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    return dimensions, spacing, origin

# Hàm để đọc thông số từ file DICOM Segmentation
def get_dicom_seg_info(dicom_file):
    image = sitk.ReadImage(dicom_file)
    dimensions = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    return dimensions, spacing, origin

# Hàm để đọc thông số từ file NIfTI
def get_nifti_info(nifti_file):
    nifti_image = nib.load(nifti_file)
    dimensions = nifti_image.shape
    spacing = nifti_image.header.get_zooms()
    origin = nifti_image.affine[:3, 3]
    return dimensions, spacing, origin

# Hàm để chuyển Origin từ RAS sang LPS
def convert_ras_to_lps(origin):
    return [-origin[0], -origin[1], origin[2]]

# Hàm để so sánh hai giá trị với ngưỡng sai lệch nhỏ
def compare_values(value1, value2, tolerance=1e-4):
    return np.allclose(value1, value2, atol=tolerance)

# Duyệt qua từng folder bệnh nhân trong DICOM
for patient_folder in os.listdir(dicom_base_path):
    if patient_folder.startswith("R01-"):
        patient_id = patient_folder.split("-")[1]

        # Kiểm tra CT
        nifti_ct_file = os.path.join(nifti_ct_path, f"lung_{patient_id}_0000.nii.gz")
        if os.path.exists(nifti_ct_file):
            patient_path = os.path.join(dicom_base_path, patient_folder)
            for random_folder in os.listdir(patient_path):
                random_folder_path = os.path.join(patient_path, random_folder)
                if os.path.isdir(random_folder_path):
                    dicom_folders = [os.path.join(random_folder_path, f) for f in os.listdir(random_folder_path) if os.path.isdir(os.path.join(random_folder_path, f))]
                    for dicom_folder in dicom_folders:
                        dicom_files = os.listdir(dicom_folder)

                        # Nếu folder chứa nhiều file DICOM -> CT
                        if len(dicom_files) > 1:
                            dicom_ct_dimensions, dicom_ct_spacing, dicom_ct_origin = get_dicom_ct_info(dicom_folder)
                            nifti_ct_dimensions, nifti_ct_spacing, nifti_ct_origin = get_nifti_info(nifti_ct_file)
                            nifti_ct_origin_lps = convert_ras_to_lps(nifti_ct_origin)

                            if dicom_ct_dimensions != nifti_ct_dimensions:
                                print(f"❌ CT Dimension mismatch for patient {patient_id}: DICOM={dicom_ct_dimensions}, NIfTI={nifti_ct_dimensions}")

                            if not compare_values(dicom_ct_spacing, nifti_ct_spacing):
                                print(f"❌ CT Spacing mismatch for patient {patient_id}: DICOM={dicom_ct_spacing}, NIfTI={nifti_ct_spacing}")

                            if not compare_values(dicom_ct_origin, nifti_ct_origin_lps):
                                print(f"❌ CT Origin mismatch for patient {patient_id}: DICOM={dicom_ct_origin}, NIfTI={nifti_ct_origin_lps}")

        # Kiểm tra Segmentation
        nifti_seg_file = os.path.join(nifti_seg_path, f"lung_{patient_id}.nii.gz")
        if os.path.exists(nifti_seg_file):
            patient_path = os.path.join(dicom_base_path, patient_folder)
            for random_folder in os.listdir(patient_path):
                random_folder_path = os.path.join(patient_path, random_folder)
                if os.path.isdir(random_folder_path):
                    dicom_folders = [os.path.join(random_folder_path, f) for f in os.listdir(random_folder_path) if os.path.isdir(os.path.join(random_folder_path, f))]
                    for dicom_folder in dicom_folders:
                        dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")]

                        # Nếu folder chứa file DICOM Segmentation
                        if len(dicom_files) == 1:
                            dicom_seg_dimensions, dicom_seg_spacing, dicom_seg_origin = get_dicom_seg_info(dicom_files[0])
                            nifti_seg_dimensions, nifti_seg_spacing, nifti_seg_origin = get_nifti_info(nifti_seg_file)
                            nifti_seg_origin_lps = convert_ras_to_lps(nifti_seg_origin)

                            if dicom_seg_dimensions != nifti_seg_dimensions:
                                print(f"❌ Segmentation Dimension mismatch for patient {patient_id}: DICOM={dicom_seg_dimensions}, NIfTI={nifti_seg_dimensions}")

                            if not compare_values(dicom_seg_spacing, nifti_seg_spacing):
                                print(f"❌ Segmentation Spacing mismatch for patient {patient_id}: DICOM={dicom_seg_spacing}, NIfTI={nifti_seg_spacing}")

                            if not compare_values(dicom_seg_origin, nifti_seg_origin_lps):
                                print(f"❌ Segmentation Origin mismatch for patient {patient_id}: DICOM={dicom_seg_origin}, NIfTI={nifti_seg_origin_lps}")
