import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import csv
from pathlib import Path

def extract_tumor_features(ct_path, seg_path):
    ct_image = sitk.ReadImage(ct_path)
    tumor_mask = sitk.ReadImage(seg_path)
    
    binary_mask = sitk.BinaryThreshold(tumor_mask, 1, 255, 1, 0)
    
    patient_id = os.path.splitext(os.path.basename(seg_path))[0]
    
    features = {"PatientID": patient_id}

    # Statistics features
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(ct_image, binary_mask)
    
    if stats_filter.HasLabel(1):
          # Volume (number of voxels)
        features["Mean"] = stats_filter.GetMean(1)     # Mean
        features["Std"] = stats_filter.GetSigma(1)     # Standard deviation
        features["Min"] = stats_filter.GetMinimum(1)   # Minimum
        features["Max"] = stats_filter.GetMaximum(1)   # Maximum
        features["Median"] = stats_filter.GetMedian(1) # Median
    else:
        print(f"Warning: No label 1 found in mask for {patient_id}")
        return None

    # Shape features
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(binary_mask)
    features["Volume"] = shape_filter.GetPhysicalSize(1)
    features["SurfaceArea"] = shape_filter.GetPerimeter(1)        # Surface area
    features["Elongation"] = shape_filter.GetElongation(1)        # Elongation
    features["Flatness"] = shape_filter.GetFlatness(1)            # Flatness
    features["Roundness"] = shape_filter.GetRoundness(1)          # Roundness
    features["EquivalentEllipsoidDiameter"] = shape_filter.GetEquivalentEllipsoidDiameter(1)  # Equivalent ellipsoid diameter
    features["Centroid"] = shape_filter.GetCentroid(1)            # Centroid
    
    try:
        glcm_filter = sitk.ScalarImageToTextureFeaturesFilter()
        glcm_filter.Execute(ct_image, binary_mask)
        
        texture_features = [
            "Energy", "Entropy", "Correlation", "InverseDifferenceMoment",
            "Inertia", "ClusterShade", "ClusterProminence"
        ]
        
        for feature in texture_features:
            try:
                features[f"GLCM_{feature}"] = glcm_filter.GetFeature(feature)
            except:
                features[f"GLCM_{feature}"] = np.nan
                
    except Exception as e:
        print(f"Warning: No texture features extracted for {patient_id}: {e}")

    return features

def process_folder(ct_folders, seg_folders, output_csv):
    all_features = []

    # Merge all segmentation files from both folders
    seg_files = []
    for seg_folder in seg_folders:
        seg_files += [os.path.join(seg_folder, f) for f in os.listdir(seg_folder) if f.endswith(('.nii', '.nii.gz', '.nrrd'))]

    # Process each segmentation file
    for seg_path in seg_files:
        seg_file = os.path.basename(seg_path)
        patient_id = seg_file.split('.')[0]

        # Find corresponding CT file for segmentation, following the *_000 format
        ct_file = None
        for ct_folder in ct_folders:
            candidate_file = os.path.join(ct_folder, f"{patient_id}_0000.nii.gz")
            if os.path.exists(candidate_file):
                ct_file = candidate_file
                break

        if ct_file is None:
            print(f"Warning: No CT found for patient {patient_id}")
            continue

        print(f"Processing patient {patient_id}")
        print(f"   - CT:  {ct_file}")
        print(f"   - SEG: {seg_path}")

        features = extract_tumor_features(ct_file, seg_path)

        if features:
            features["PatientID"] = patient_id
            all_features.append(features)

    # Save results
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved features to {output_csv}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    
    ct_folders = ["../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/imagesTr", "../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/imagesTs"]
    seg_folders = ["../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTr", "../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTs"]
    output_csv = "tumor_features_2.csv"

    
    process_folder(ct_folders, seg_folders, output_csv)