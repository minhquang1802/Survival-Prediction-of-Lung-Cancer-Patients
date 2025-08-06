import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def extract_features_both_methods(image_path, mask_path, patient_id):
    """
    Feature extraction using both SimpleITK and PyRadiomics for a given patient.
    
    Parameters:
    -----------
    image_path: str
        Path to the CT image
    mask_path: str
        Path to the segmentation mask
    patient_id: str
        Patient ID
        
    Returns:
    --------
    dict
        Dictionary containing features extracted from both libraries
    """
    try:
        # Read image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        # Ensure mask is binary
        binary_mask = sitk.Equal(mask, 1)
        binary_mask = sitk.Cast(binary_mask, sitk.sitkUInt8)

        # Dictionary containing results
        features = {"PatientID": patient_id}
        
        # ================== 1. SimpleITK ==================

        # Extract shape features
        shape_filter = sitk.LabelShapeStatisticsImageFilter()
        shape_filter.Execute(binary_mask)

        # Save SimpleITK features to dictionary
        features["SimpleITK_Volume"] = shape_filter.GetPhysicalSize(1)
        features["SimpleITK_SurfaceArea"] = shape_filter.GetPerimeter(1)
        features["SimpleITK_Elongation"] = shape_filter.GetElongation(1)
        features["SimpleITK_Flatness"] = shape_filter.GetFlatness(1)
        
        # Check if GetRoundness is available in the SimpleITK version
        try:
            features["SimpleITK_Roundness"] = shape_filter.GetRoundness(1)
        except AttributeError:
            # If GetRoundness is not available, approximate using Elongation and Flatness
            max_axis = max(shape_filter.GetElongation(1), shape_filter.GetFlatness(1))
            features["SimpleITK_Roundness"] = 1.0 / max_axis if max_axis != 0 else 0
        
        # ================== 2. PyRadiomics ==================
        
        # Configure PyRadiomics feature extractor
        # settings = {}
        # settings['binWidth'] = 25
        # settings['resampledPixelSpacing'] = None
        # settings['interpolator'] = sitk.sitkBSpline
        # settings['verbose'] = False
        
        # Enable only shape features
        # settings['enabledFeatures'] = {'shape'}
        
        # Extract features
        extractor = featureextractor.RadiomicsFeatureExtractor()
        pyrad_features = extractor.execute(image, binary_mask)
        
        # Save PyRadiomics features to dictionary
        features["PyRadiomics_VoxelVolume"] = pyrad_features.get('original_shape_VoxelVolume', None)
        features["PyRadiomics_SurfaceArea"] = pyrad_features.get('original_shape_SurfaceArea', None)
        features["PyRadiomics_Elongation"] = pyrad_features.get('original_shape_Elongation', None)
        features["PyRadiomics_Flatness"] = pyrad_features.get('original_shape_Flatness', None)
        features["PyRadiomics_Sphericity"] = pyrad_features.get('original_shape_Sphericity', None)
        
        return features
    
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        return {"PatientID": patient_id, "Error": str(e)}

def calculate_feature_discrepancy(features_df):
    """
    Calculate the discrepancy between values extracted from SimpleITK and PyRadiomics
    
    Parameters:
    -----------
    features_df: pandas.DataFrame
        DataFrame containing extracted features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated discrepancies
    """
    # Create a copy of the DataFrame
    df = features_df.copy()
    
    # Calculate relative discrepancy for each feature
    # Note: Volume uses different units in SimpleITK and PyRadiomics,
    # so we calculate ratio instead of absolute difference
    
    # Calculate relative discrepancy for Volume (ratio)
    df['Volume_Discrepancy'] = abs(df['SimpleITK_Volume'] / df['PyRadiomics_VoxelVolume'] - 1)
    
    # For remaining features, calculate absolute discrepancy
    df['SurfaceArea_Discrepancy'] = abs(df['SimpleITK_SurfaceArea'] - df['PyRadiomics_SurfaceArea'])
    df['Elongation_Discrepancy'] = abs(df['SimpleITK_Elongation'] - df['PyRadiomics_Elongation'])
    df['Flatness_Discrepancy'] = abs(df['SimpleITK_Flatness'] - df['PyRadiomics_Flatness'])
    df['Roundness_Discrepancy'] = abs(df['SimpleITK_Roundness'] - df['PyRadiomics_Sphericity'])
    
    # Calculate normalized total discrepancy
    # Normalize discrepancies to ensure fairness between features
    for col in ['Volume_Discrepancy', 'SurfaceArea_Discrepancy', 'Elongation_Discrepancy', 
                'Flatness_Discrepancy', 'Roundness_Discrepancy']:
        if df[col].max() != df[col].min():
            df[col + '_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col + '_normalized'] = 0
    
    # Calculate normalized total discrepancy
    df['Total_Discrepancy'] = (
        df['Volume_Discrepancy_normalized'] + 
        df['SurfaceArea_Discrepancy_normalized'] + 
        df['Elongation_Discrepancy_normalized'] + 
        df['Flatness_Discrepancy_normalized'] + 
        df['Roundness_Discrepancy_normalized']
    )
    
    return df

def main(ct_folder, segmentation_folder, output_folder):
    """
    Main function to extract features and find 10 patients with the least discrepancy
    
    Parameters:
    -----------
    ct_folder: str
        Path to folder containing CT images
    segmentation_folder: str
        Path to folder containing segmentation masks
    output_folder: str
        Path to output folder to save results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CT files
    ct_files = sorted(glob.glob(os.path.join(ct_folder, "lung_*_0000.nii.gz")))
    
    # Check if any CT files were found
    if not ct_files:
        print(f"No CT files found in folder {ct_folder}")
        return
    
    print(f"Found {len(ct_files)} CT files")
    
    # Create list to store results
    all_features = []
    
    # Extract features for each patient
    for ct_file in tqdm(ct_files, desc="Feature extraction"):
        # Determine patient ID from filename
        base_filename = os.path.basename(ct_file)
        patient_id = base_filename.split("_0000")[0]  # "lung_XXX"
        
        # Determine path to corresponding segmentation file
        seg_file = os.path.join(segmentation_folder, f"{patient_id}.nii.gz")
        
        # Check if segmentation file exists
        if not os.path.exists(seg_file):
            print(f"Segmentation file not found for patient {patient_id}")
            continue
        
        # Extract features
        features = extract_features_both_methods(ct_file, seg_file, patient_id)
        
        # Add to results list
        all_features.append(features)
    
    # Check if there's any valid data
    if not all_features:
        print("No valid data after feature extraction")
        return
    
    # Convert results list to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Remove records with errors
    features_df = features_df[~features_df.filter(like='Error').any(axis=1)]
    
    # Save original feature data
    features_df.to_csv(os.path.join(output_folder, "all_features.csv"), index=False)
    
    # Calculate discrepancy between SimpleITK and PyRadiomics
    discrepancy_df = calculate_feature_discrepancy(features_df)
    
    # Save discrepancy data
    discrepancy_df.to_csv(os.path.join(output_folder, "feature_discrepancy.csv"), index=False)
    
    # Sort by total discrepancy in ascending order
    sorted_df = discrepancy_df.sort_values(by='Total_Discrepancy')
    
    # Get 10 patients with the least discrepancy
    top10_patients = sorted_df.head(10)
    
    # Save top 10 results
    top10_patients.to_csv(os.path.join(output_folder, "top10_least_discrepancy.csv"), index=False)
    
    # Plot comparison charts for 10 patients with the least discrepancy
    # create_comparison_plots(top10_patients, output_folder)
    
    print(f"Results have been saved in folder {output_folder}")
    print("10 patients with the least discrepancy:")
    print(top10_patients[['PatientID', 'Total_Discrepancy']])
    
    return top10_patients

if __name__ == "__main__":
    import argparse
    
    ct_folder = "../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/imagesTr"
    seg_folder = "../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTr"
    output_folder = "../../../FeatureExtraction/image_branch/method_1/output"

    main(ct_folder, seg_folder, output_folder)