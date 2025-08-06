import os
import nibabel as nib
import numpy as np

# Path
base_path = "E:/DATN_LVTN/dataset/Dataset015_lungTumor"
ct_path = os.path.join(base_path, "CT")
seg_path = os.path.join(base_path, "Segmentation")

for ct_file in os.listdir(ct_path):
    if ct_file.endswith(".nii.gz"):
        # Get ID
        patient_id = ct_file.replace("lung_", "").replace("_0000.nii.gz", "")
        
        # Create path
        ct_file_path = os.path.join(ct_path, ct_file)
        seg_file_path = os.path.join(seg_path, f"lung_{patient_id}.nii.gz")
        
        # Check Segmentation file exist
        if os.path.exists(seg_file_path):
            ct_image = nib.load(ct_file_path)
            seg_image = nib.load(seg_file_path)
            
            # Get Dimension, Spacing, and Origin
            ct_data = ct_image.get_fdata()
            seg_data = seg_image.get_fdata()
            
            ct_affine = ct_image.affine
            seg_affine = seg_image.affine
            
            ct_spacing = ct_image.header.get_zooms()
            seg_spacing = seg_image.header.get_zooms()
            
            ct_origin = ct_affine[:3, 3]
            seg_origin = seg_affine[:3, 3]
            
            # Check Origin
            if not np.allclose(ct_origin, seg_origin):
                print(f"Origin mismatch for patient {patient_id}: CT={ct_origin}, Segmentation={seg_origin}")
                seg_image = nib.Nifti1Image(seg_data, ct_affine)
                nib.save(seg_image, seg_file_path)
                print(f"Origin fixed for patient {patient_id}")
            
            # Check Spacing
            if not np.allclose(ct_spacing, seg_spacing):
                print(f"Spacing mismatch for patient {patient_id}: CT={ct_spacing}, Segmentation={seg_spacing}")
                fixed_seg_image = nib.Nifti1Image(seg_data, seg_affine)
                fixed_seg_image.header.set_zooms(ct_spacing)
                nib.save(fixed_seg_image, seg_file_path)
                print(f"Spacing fixed for patient {patient_id}")
            
            # Check dimension
            if ct_data.shape != seg_data.shape:
                print(f"Dimension mismatch for patient {patient_id}: CT={ct_data.shape}, Segmentation={seg_data.shape}")
                
                # Get Z Coords
                ct_z_coords = [ct_origin[2] + i * ct_spacing[2] for i in range(ct_data.shape[2])]
                seg_z_coords = [seg_origin[2] + i * seg_spacing[2] for i in range(seg_data.shape[2])]

                try:
                    start_index = ct_z_coords.index(min(ct_z_coords, key=lambda x: abs(x - seg_z_coords[0])))
                    end_index = ct_z_coords.index(min(ct_z_coords, key=lambda x: abs(x - seg_z_coords[-1])))
                except ValueError as e:
                    print(f"Error processing patient {patient_id}: {e}")
                    continue

                # Create new segmentation mask
                new_seg_data = np.zeros(ct_data.shape, dtype=np.uint8)
                new_seg_data[:, :, start_index:end_index + 1] = seg_data

                new_seg_image = nib.Nifti1Image(new_seg_data, ct_affine)
                new_seg_file_path = os.path.join(seg_path, f"lung_{patient_id}_fixed.nii.gz")
                nib.save(new_seg_image, new_seg_file_path)
                print(f"New segmentation mask saved for patient {patient_id}: {new_seg_file_path}")
