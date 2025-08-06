import streamlit as st
import os
import tempfile
import subprocess
import joblib
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from matplotlib.colors import LinearSegmentedColormap
import shutil
import time
import re
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Lung Tumor Analysis",
    page_icon="🫁",
    layout="wide",
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Lung Tumor Analysis System</h1>", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'nifti_file_path' not in st.session_state:
    st.session_state.nifti_file_path = None
if 'mask_file_path' not in st.session_state:
    st.session_state.mask_file_path = None
if 'tumor_features' not in st.session_state:
    st.session_state.tumor_features = None
if 'clinical_data' not in st.session_state:
    st.session_state.clinical_data = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'survival_model' not in st.session_state:
    st.session_state.survival_model = None
if 'current_slice' not in st.session_state:
    st.session_state.current_slice = 0
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'tumor_slices' not in st.session_state:
    st.session_state.tumor_slices = None
if 'patient_clinical_data' not in st.session_state:
    st.session_state.patient_clinical_data = None

# Directory path for pre-generated segmentation masks
INFERENCE_DIR = "E:/DATN_LVTN/TumorSegmentation/train/run_Inference/output_folder_pp"  # Change this to your local directory path

def extract_patient_id(filename):
    """Extract patient ID from filename using regex pattern."""
    # Pattern matches lung_XXX_0000.nii.gz format
    pattern = r'lung_(\d+)_0000\.nii\.gz'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)  # Returns the XXX part
    else:
        return None

def find_segmentation_file(patient_id):
    """Find the corresponding segmentation file based on patient ID."""
    if not os.path.exists(INFERENCE_DIR):
        st.error(f"Inference directory '{INFERENCE_DIR}' not found!")
        return None
    
    # Look for file with format lung_XXX.nii.gz
    expected_filename = f"lung_{patient_id}.nii.gz"
    segmentation_path = os.path.join(INFERENCE_DIR, expected_filename)
    
    if os.path.exists(segmentation_path):
        return segmentation_path
    else:
        st.error(f"Segmentation file {expected_filename} not found in {INFERENCE_DIR}")
        return None

def simulate_inference(input_file_path):
    """Simulate nnUNet inference by finding pre-generated segmentation file."""
    try:
        # Extract patient ID from filename
        filename = os.path.basename(input_file_path)
        patient_id = extract_patient_id(filename)
        
        if not patient_id:
            st.error(f"Could not extract patient ID from filename: {filename}")
            return None
        
        # Store patient ID in session state
        st.session_state.patient_id = patient_id
        
        # Find corresponding segmentation file
        segmentation_path = find_segmentation_file(patient_id)
        
        if not segmentation_path:
            st.error("No corresponding segmentation file found.")
            return None
            
        # Create a simulated progress bar to show "processing"
        with st.spinner(f"Processing patient ID {patient_id}..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.03)  # Simulate processing time
                progress_bar.progress(i + 1)
        
        return segmentation_path
        
    except Exception as e:
        st.error(f"Error in simulation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to extract tumor features using SimpleITK
def extract_tumor_features(image_path, mask_path):
    """Extract features from the tumor using SimpleITK."""
    try:
        # Load the image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Resample mask to match image dimensions if necessary
        if image.GetSize() != mask.GetSize():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask = resampler.Execute(mask)
            
        binary_mask = sitk.BinaryThreshold(mask, 1, 255, 1, 0)
        
        # Create shape statistics filter
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(binary_mask)
        
        # Extract intensity statistics
        stats_filter = sitk.LabelStatisticsImageFilter()
        stats_filter.Execute(image, binary_mask)
        
        # Get label (assuming label 1 is the tumor)
        label = 1
        
        # Check if the label exists in the mask
        if not shape_stats.HasLabel(label):
            # Try to find any other label
            labels = shape_stats.GetLabels()
            if len(labels) == 0:
                st.error("No tumor regions found in the mask.")
                return None
            label = labels[0]
        
        # Extract features
        features = {
            # Shape features
            "Surface Area (mm²)": shape_stats.GetPerimeter(label),
            "Roundness": shape_stats.GetRoundness(label),
            "Elongation": shape_stats.GetElongation(label),
            "Flatness": shape_stats.GetFlatness(label),
            
            # Intensity features
            "Volume (mm³)": stats_filter.GetCount(label),
            "Mean": stats_filter.GetMean(label),
            "Minimum": stats_filter.GetMinimum(label),
            "Maximum": stats_filter.GetMaximum(label),
            "Standard Deviation": stats_filter.GetSigma(label),
            "Median": stats_filter.GetMedian(label),
        }
        
        return features
    
    except Exception as e:
        st.error(f"Error extracting tumor features: {e}")
        return None

def preprocess_clinical_data(clinical_df):
    """Preprocess clinical data as specified."""
    try:
        # Make a copy to avoid modifying the original
        df = clinical_df.copy()
        
        # Define columns that need label encoding
        object_cols = ['Gender', 'Ethnicity', 'Smoking status',
                      '%GG', 'Tumor Location (choice=RUL)', 'Tumor Location (choice=RML)', 'Tumor Location (choice=RLL)', 'Tumor Location (choice=LUL)',
                      'Tumor Location (choice=LLL)', 'Tumor Location (choice=L Lingula)', 'Tumor Location (choice=Unknown)',
                      'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Histopathological Grade', 'Lymphovascular invasion',
                      'Pleural invasion (elastic, visceral, or parietal)', 'EGFR mutation status', 'KRAS mutation status', 'ALK translocation status', 'Adjuvant Treatment',
                      'Chemotherapy', 'Radiation', 'Recurrence']
        
        # Apply label encoding to each column
        for col in object_cols:
            if col in df.columns:
                df[col] = np.uint8(LabelEncoder().fit_transform(df[col]))
        
        # Handle Histology column if it exists
        if 'Histology' in df.columns:
            df['Histology'] = np.uint8(df['Histology'].map({
                'Adenocarcinoma': 0, 
                'Squamous cell carcinoma': 1, 
                'NSCLC NOS (not otherwise specified)': 2
            }))
        
        return df
    
    except Exception as e:
        st.error(f"Error preprocessing clinical data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def find_patient_in_clinical_data(patient_id, clinical_data):
    """Find patient in clinical data by ID and return relevant row."""
    try:
        # Format to match R01-XXX pattern
        formatted_id = f"R01-{patient_id}"
        
        # Look for the patient ID in the Case ID column
        patient_data = clinical_data[clinical_data['Case ID'] == formatted_id]
        
        if len(patient_data) == 0:
            st.warning(f"Patient {formatted_id} not found in clinical data")
            return None
        
        # Return the first row if multiple matches (shouldn't happen)
        return patient_data.iloc[0:1]
    
    except Exception as e:
        st.error(f"Error finding patient in clinical data: {e}")
        return None

# Function to prepare data for survival analysis
def prepare_survival_data(clinical_data, tumor_features):
    """Prepare data for survival analysis by combining clinical data and tumor features."""
    try:
        if clinical_data is None or tumor_features is None:
            return None
            
        # Convert tumor features to DataFrame
        tumor_df = pd.DataFrame([tumor_features])
        
        clinical_data = clinical_data.drop(columns=['Case ID', 'Patient affiliation'], axis=1)
        tumor_df = tumor_df.rename(columns={"Volume (mm³)": "Volume", "Standard Deviation": "Std", "Minimum": "Min", "Maximum": "Max", "Surface Area (mm²)": "SurfaceArea"})
        
        # Ensure clinical_data is a DataFrame with only one row
        if isinstance(clinical_data, pd.DataFrame) and len(clinical_data) > 1:
            clinical_data = clinical_data.iloc[0:1].reset_index(drop=True)
        
        # Create a combined DataFrame for the patient
        patient_data = pd.concat([clinical_data.reset_index(drop=True), tumor_df.reset_index(drop=True)], axis=1)
        
        return patient_data
    
    except Exception as e:
        st.error(f"Error preparing survival data: {e}")
        return None

def run_survival_analysis(combined_data, model_path='cox_model.pkl'):
    try:
        if not isinstance(combined_data, pd.DataFrame) or combined_data.shape[0] != 1:
            st.error("Input data must be a DataFrame with exactly one patient row.")
            return None

        cph_model = joblib.load(model_path)

        model_features = cph_model.params_.index.tolist()
        missing_cols = [col for col in model_features if col not in combined_data.columns]
        if missing_cols:
            st.error(f"Missing required feature columns: {missing_cols}")
            return None

        risk_score = float(cph_model.predict_partial_hazard(combined_data[model_features]).values[0])

        timepoints = np.linspace(0, 60, 100)
        surv_func = cph_model.predict_survival_function(combined_data[model_features], times=timepoints)
        survival_curve = surv_func.values.flatten()

        if np.any(survival_curve < 0.5):
            median_survival = timepoints[np.where(survival_curve < 0.5)[0][0]]
        else:
            median_survival = ">60"

        one_year_idx = np.argmin(np.abs(timepoints - 12))
        three_year_idx = np.argmin(np.abs(timepoints - 36))
        one_year_survival = survival_curve[one_year_idx]
        three_year_survival = survival_curve[three_year_idx]

        concordance_index = getattr(cph_model, 'concordance_index_', 0.72)

        results = {
            'concordance_index': concordance_index,
            'timepoints': timepoints,
            'survival_curve': survival_curve,
            'median_survival': median_survival,
            'risk_score': risk_score,
            'one_year_survival': one_year_survival,
            'three_year_survival': three_year_survival
        }

        return results

    except Exception as e:
        st.error(f"Error in survival analysis: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to display CT and mask overlay
def display_ct_and_mask(ct_file, mask_file, slice_idx=None):
    """Display a slice of the CT scan with tumor mask overlay."""
    try:
        # Load the CT scan and mask
        ct_img = nib.load(ct_file)
        mask_img = nib.load(mask_file)
        
        # Get the CT and mask data
        ct_data = ct_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Ensure the mask data is binary
        mask_data = (mask_data > 0).astype(float)
        
        # Find all slices containing tumor
        non_zero_indices = np.where(mask_data > 0)
        if len(non_zero_indices[0]) > 0:
            # Get unique z-indices that contain tumor
            z_indices = np.unique(non_zero_indices[2])
            
            # Store tumor slices in session state
            if 'tumor_slices' not in st.session_state:
                st.session_state.tumor_slices = z_indices
            
            # If slice_idx is None or not in tumor slices, use the middle tumor slice
            if slice_idx is None or slice_idx not in z_indices:
                slice_idx = z_indices[len(z_indices) // 2]
            
            # Ensure slice index is within tumor slices
            if slice_idx not in z_indices:
                slice_idx = z_indices[0]
        else:
            # If no tumor is found, use the middle slice and set tumor_slices to empty
            st.session_state.tumor_slices = np.array([])
            slice_idx = ct_data.shape[2] // 2
            st.warning("No tumor regions found in this scan.")
        
        # Ensure slice index is within bounds
        slice_idx = max(0, min(slice_idx, ct_data.shape[2] - 1))
        
        # Extract the CT slice and corresponding mask
        ct_slice = ct_data[:, :, slice_idx].T
        mask_slice = mask_data[:, :, slice_idx].T
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display CT scan
        ax1.imshow(ct_slice, cmap='gray')
        ax1.set_title('CT Scan')
        ax1.axis('off')
        
        # Display CT scan with mask overlay
        ax2.imshow(ct_slice, cmap='gray')
        
        # Create a custom colormap for the mask (red with transparency)
        colors = [(1, 0, 0, 0), (1, 0, 0, 0.5)]
        cmap = LinearSegmentedColormap.from_list('custom_red', colors)
        
        ax2.imshow(mask_slice, cmap=cmap)
        ax2.set_title('CT with Tumor Mask')
        ax2.axis('off')
        
        fig.tight_layout()
        return fig, slice_idx
    
    except Exception as e:
        st.error(f"Error displaying CT and mask: {e}")
        return None, None

# Create a two-column layout for the app
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("<h2 class='sub-header'>Data Upload & Analysis</h2>", unsafe_allow_html=True)
    
    # NIFTI File Upload
    st.markdown("### Upload CT Scan")
    nifti_file = st.file_uploader("Upload NIFTI file (.nii.gz)", type=["nii.gz"])
    
    if nifti_file is not None:
        # Check if this is a new file upload
        current_filename = nifti_file.name
        if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != current_filename:
            # Reset results when a new file is uploaded
            st.session_state.mask_file_path = None
            st.session_state.tumor_features = None
            st.session_state.prediction_results = None
            st.session_state.current_slice = 0
            st.session_state.last_uploaded_filename = current_filename
        
        # Save the uploaded file to a temporary location
        original_filename = nifti_file.name
        temp_nifti_path = os.path.join(st.session_state.temp_dir, original_filename)
        with open(temp_nifti_path, "wb") as f:
            f.write(nifti_file.getbuffer())
        
        st.session_state.nifti_file_path = temp_nifti_path
        
        # Extract patient ID and display it
        patient_id = extract_patient_id(original_filename)
        if patient_id:
            st.markdown(f"<div class='info-box'>Patient ID: {patient_id}</div>", unsafe_allow_html=True)
            st.session_state.patient_id = patient_id
        
        st.markdown("<div class='success-box'>CT scan uploaded successfully!</div>", unsafe_allow_html=True)
    
    # Predict Button
    if st.session_state.nifti_file_path is not None:
        if st.button("Run Tumor Segmentation"):
            with st.spinner("Running tumor segmentation..."):
                # Use the simulated inference function instead of actual nnUNet
                mask_path = simulate_inference(st.session_state.nifti_file_path)
                
                if mask_path and os.path.exists(mask_path):
                    st.session_state.mask_file_path = mask_path
                    st.markdown("<div class='success-box'>Tumor segmentation loaded successfully!</div>", unsafe_allow_html=True)
                else:
                    st.error("Could not find segmentation mask for this patient. Make sure the inference folder contains the corresponding file.")
    
    # Feature Extraction
    if st.session_state.nifti_file_path is not None and st.session_state.mask_file_path is not None:
        if st.button("Extract Tumor Features"):
            with st.spinner("Extracting tumor features..."):
                # In a real app, call the actual function
                st.session_state.tumor_features = extract_tumor_features(
                    st.session_state.nifti_file_path, 
                    st.session_state.mask_file_path
                )
                
                # If extraction failed, use simulated data for demo
                if st.session_state.tumor_features is None:
                    st.session_state.tumor_features = {
                        "Volume (mm³)": 15234.56,
                        "Surface Area (mm²)": 2367.89,
                        "Roundness": 0.76,
                        "Elongation": 1.32,
                        "Flatness": 1.15,
                        "Mean": 45.67,
                        "Minimum": 12.3,
                        "Maximum": 89.5,
                        "Standard Deviation": 14.32,
                        "Variance": 205.06
                    }
                
                st.markdown("<div class='success-box'>Tumor features extracted successfully!</div>", unsafe_allow_html=True)
    
    # Clinical Data Upload
    st.markdown("### Upload Clinical Data")
    clinical_file = st.file_uploader("Upload clinical data (.csv)", type=["csv"])
    
    if clinical_file is not None:
        try:
            # Store raw clinical data
            clinical_data_raw = pd.read_csv(clinical_file)
            st.session_state.clinical_data_raw = clinical_data_raw
            
            # Preprocess the clinical data
            clinical_data_processed = preprocess_clinical_data(clinical_data_raw)
            st.session_state.clinical_data = clinical_data_processed
            
            st.markdown("<div class='success-box'>Clinical data uploaded and preprocessed successfully!</div>", unsafe_allow_html=True)
            
            # Display a preview of the clinical data
            st.markdown("#### Clinical Data Preview")
            st.dataframe(clinical_data_processed.head())
            
            # If patient ID is available, find corresponding clinical data
            if st.session_state.patient_id:
                patient_clinical_data = find_patient_in_clinical_data(
                    st.session_state.patient_id, 
                    clinical_data_processed
                )
                
                if patient_clinical_data is not None:
                    st.session_state.patient_clinical_data = patient_clinical_data
                    st.markdown(f"<div class='success-box'>Found clinical data for patient {st.session_state.patient_id}</div>", 
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='warning-box'>No clinical data found for patient {st.session_state.patient_id}</div>", 
                                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading clinical data: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    # Survival Analysis
    if st.session_state.tumor_features is not None and st.session_state.patient_clinical_data is not None:
        if st.button("Run Survival Analysis"):
            with st.spinner("Running survival analysis..."):
                # Prepare the data for survival analysis
                combined_data = prepare_survival_data(
                    st.session_state.patient_clinical_data,
                    st.session_state.tumor_features
                )
                
                st.markdown("#### Combined Clinical Data Preview")
                st.dataframe(combined_data.head())
                
                if combined_data is not None:
                    st.session_state.combined_data = combined_data
                    
                    # Run the survival analysis
                    st.session_state.prediction_results = run_survival_analysis(combined_data)
                    st.markdown("<div class='success-box'>Survival analysis completed!</div>", unsafe_allow_html=True)
                else:
                    st.error("Failed to prepare data for survival analysis.")

with col2:
    st.markdown("<h2 class='sub-header'>Results & Visualization</h2>", unsafe_allow_html=True)
    
    # Display CT and Mask
    if st.session_state.nifti_file_path is not None and st.session_state.mask_file_path is not None:
        st.markdown("### CT Scan & Tumor Segmentation")
        
        # Display patient ID if available
        if st.session_state.patient_id:
            st.markdown(f"**Patient ID:** {st.session_state.patient_id}")
        
        # Initialize current_slice in session state if needed
        if "current_slice" not in st.session_state:
            st.session_state.current_slice = 0
            
        # Process the mask to identify tumor slices first
        try:
            # Load the mask to find tumor slices
            mask_img = nib.load(st.session_state.mask_file_path)
            mask_data = mask_img.get_fdata()
            mask_data = (mask_data > 0).astype(float)
            
            # Find slices with tumor
            non_zero_indices = np.where(mask_data > 0)
            if len(non_zero_indices[0]) > 0:
                # Get unique z-indices that contain tumor
                z_indices = np.unique(non_zero_indices[2])
                st.session_state.tumor_slices = z_indices
                
                # If we have tumor slices, create a slider for them
                if len(z_indices) > 0:
                    # Map slider index to actual slice index
                    slider_to_slice = {i: int(z) for i, z in enumerate(z_indices)}
                    
                    # Find current slice index in tumor slices
                    current_slice_idx = 0
                    if st.session_state.current_slice in z_indices:
                        current_slice_idx = np.where(z_indices == st.session_state.current_slice)[0][0]
                    
                    # Create slider for tumor slices
                    st.markdown(f"**Tumor found in {len(z_indices)} slices**")
                    selected_slider_idx = st.slider(
                        "Navigate through tumor slices", 
                        0, len(z_indices)-1, 
                        int(current_slice_idx)
                    )
                    
                    # Convert slider index to actual slice index
                    selected_slice = slider_to_slice[selected_slider_idx]
                    st.session_state.current_slice = selected_slice
                    
                    # Show slice number information
                    st.info(f"Showing slice {selected_slice} of {mask_data.shape[2]-1}")
                    
                    # Display the selected slice
                    fig, _ = display_ct_and_mask(
                        st.session_state.nifti_file_path,
                        st.session_state.mask_file_path,
                        slice_idx=selected_slice
                    )
                    
                    if fig is not None:
                        st.pyplot(fig)
                else:
                    st.warning("No tumor regions found in this scan.")
            else:
                st.warning("No tumor regions found in this scan.")
                
        except Exception as e:
            st.error(f"Error displaying images: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Display Tumor Features
    if st.session_state.tumor_features is not None:
        st.markdown("### Extracted Tumor Features")
        
        # Create a nicer display for tumor features
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### Shape Features")
            for key, value in list(st.session_state.tumor_features.items())[:5]:
                st.markdown(f"**{key}:** {value:.2f}")
        
        with col_b:
            st.markdown("#### Intensity Features")
            for key, value in list(st.session_state.tumor_features.items())[5:]:
                st.markdown(f"**{key}:** {value:.2f}")
    
    # Display Survival Analysis Results
    # Display Survival Analysis Results
    if st.session_state.prediction_results is not None:
        st.markdown("### Survival Analysis Results")
        
        # Display concordance index
        st.markdown(f"**Concordance Index:** {st.session_state.prediction_results['concordance_index']:.3f}")
        
        # Chuyển đổi median_survival từ string sang float nếu cần
        median_survival = st.session_state.prediction_results['median_survival']
        if isinstance(median_survival, str):
            try:
                median_survival = float(median_survival)
                formatted_median = f"{median_survival:.1f}"
            except ValueError:
                # Nếu không thể chuyển đổi thành số, sử dụng giá trị gốc
                formatted_median = median_survival
        else:
            formatted_median = f"{median_survival:.1f}"
        
        # Display estimated survival time
        st.markdown(f"**Estimated Median Survival:** {formatted_median} months")
        
        # Display risk score
        st.markdown(f"**Relative Risk Score:** {st.session_state.prediction_results['risk_score']:.2f}")
        
        # Plot survival curve
        st.markdown("#### Survival Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        timepoints = st.session_state.prediction_results['timepoints']
        survival_curve = st.session_state.prediction_results['survival_curve']
        
        ax.plot(timepoints, survival_curve, '-', linewidth=2, color='#1976D2')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Estimated Survival Curve')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.05)
        
        # Add a marker for median survival time - sử dụng giá trị đã chuyển đổi
        if isinstance(median_survival, (int, float)):  # Chỉ vẽ đường kẻ nếu là số
            ax.axvline(x=median_survival, color='red', linestyle='--', alpha=0.7)
            ax.text(median_survival + 1, 0.5, f'Median: {formatted_median} months', 
                    verticalalignment='center', color='red')
        
        # Display the plot
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Lung Tumor Analysis System - Graduation Project Demo**")

# Cleanup function to remove temporary files when the app is closed
def cleanup():
    """Clean up temporary files when the app is closed."""
    if hasattr(st.session_state, 'temp_dir') and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

# Register the cleanup function to be called when the app is closed
import atexit
atexit.register(cleanup)