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
        features["Volume"] = stats_filter.GetCount(1)  # Volume (number of voxels)
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

def process_folder(ct_folder, seg_folder, output_csv):
    all_features = []
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith(('.nii', '.nii.gz', '.nrrd'))]
    
    for seg_file in seg_files:
        patient_id = seg_file.split('.')[0]
        ct_id = f"{patient_id}_000"
        
        ct_files = os.listdir(ct_folder)
        
        ct_file = None
        for file in ct_files:
            if file.startswith(ct_id):
                ct_file = os.path.join(ct_folder, file)
                break
        
        if ct_file is None:
            print(f"Warning: No CT found for patient {patient_id} (looking for {ct_id})")
            continue
            
        seg_path = os.path.join(seg_folder, seg_file)

        print(f"Processing patient {patient_id}...")
        print(f"  - CT: {ct_file}")
        print(f"  - Segmentation: {seg_path}")
        
        features = extract_tumor_features(ct_file, seg_path)
        
        if features:
            all_features.append(features)
    
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"Saved features to {output_csv}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    
    ct_path = "../../../FeatureExtraction/image_branch/dataset/ct"
    mask_path = "../../../FeatureExtraction/image_branch/dataset/segmentation_mask"
    output_path = 'tumor_features.csv'
    
    process_folder(ct_path, mask_path, output_path)