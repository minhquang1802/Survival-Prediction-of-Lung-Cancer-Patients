import os
import pydicom
import nibabel as nib
import numpy as np
import SimpleITK as sitk

# path
raw_dataset_path = "D:/Books/DATN_LVTN/dataset/NSCLC_Radiogenomics/manifest-1732266091299/NSCLC Radiogenomics"
output_dataset_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor"
log_file = "E:/DATN_LVTN/nnUNet/log/missing_slicethickness.txt"
ct_output_path = os.path.join(output_dataset_path, "CT")
seg_output_path = os.path.join(output_dataset_path, "Segmentation")

os.makedirs(ct_output_path, exist_ok=True)
os.makedirs(seg_output_path, exist_ok=True)

# Chuyển đổi CT qua NIfTI
def convert_ct_to_nifti(dicom_folder, output_file):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    sitk.WriteImage(image, output_file)
    print(f"Converted CT to {output_file}")

# Chuyển đổi Segmentation qua NIfTI
def convert_segmentation_to_nifti(dicom_file, output_file, patient_id):
    dcm = pydicom.dcmread(dicom_file)
    slice_thickness = None

    # Shared Functional Groups Sequence lấy Slice Thickness
    shared_sequence = dcm.get((0x5200, 0x9229), None)
    if shared_sequence is not None:
        shared_sequence_value = shared_sequence.value
        if len(shared_sequence_value) > 0:
            pixel_measures_sequence = shared_sequence_value[0].get((0x0028, 0x9110), None)
            if pixel_measures_sequence is not None:
                pixel_measures_value = pixel_measures_sequence.value
                if len(pixel_measures_value) > 0:
                    slice_thickness_element = pixel_measures_value[0].get((0x0018, 0x0050), None)
                    if slice_thickness_element is not None:
                        slice_thickness = float(slice_thickness_element.value)

    # Nếu không tìm thấy Slice Thickness, đặt giá trị mặc định
    if slice_thickness is None:
        print(f"Slice Thickness not found for patient {patient_id}. Using default value: 1.0")
        slice_thickness = 1.0
        with open(log_file, "a") as log:
            log.write(f"Patient ID: {patient_id} - Missing Slice Thickness\n")
    else:
        print(f"Slice Thickness for patient {patient_id}: {slice_thickness}")

    image = sitk.ReadImage(dicom_file)

    # Kiểm tra BitsAllocated để đảm bảo dữ liệu đúng định dạng
    if image.GetPixelIDValue() != sitk.sitkUInt8:
        image = sitk.Cast(image, sitk.sitkUInt8)

    # Sửa lại spacing để đồng bộ Slice Thickness
    spacing = list(image.GetSpacing())
    spacing[2] = slice_thickness
    image.SetSpacing(spacing)

    sitk.WriteImage(image, output_file)
    print(f"Converted Segmentation to {output_file} (with corrected slice thickness: {slice_thickness})")

for patient_folder in os.listdir(raw_dataset_path):
    patient_path = os.path.join(raw_dataset_path, patient_folder)

    if os.path.isdir(patient_path) and patient_folder.startswith("R01-"):
        patient_id = patient_folder.split("-")[1]

        for random_folder in os.listdir(patient_path):
            random_folder_path = os.path.join(patient_path, random_folder)

            if os.path.isdir(random_folder_path):
                dicom_folders = [os.path.join(random_folder_path, f) for f in os.listdir(random_folder_path) if os.path.isdir(os.path.join(random_folder_path, f))]

                for dicom_folder in dicom_folders:
                    dicom_files = os.listdir(dicom_folder)

                    # folder CT
                    if len(dicom_files) > 1:
                        ct_output_file = os.path.join(ct_output_path, f"lung_{patient_id}_0000.nii.gz")
                        convert_ct_to_nifti(dicom_folder, ct_output_file)

                    # folder Segmentation
                    elif len(dicom_files) == 1:
                        seg_output_file = os.path.join(seg_output_path, f"lung_{patient_id}.nii.gz")
                        convert_segmentation_to_nifti(os.path.join(dicom_folder, dicom_files[0]), seg_output_file, patient_id)

print("Processing completed.")
