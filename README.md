# Survival Prediction of Lung Cancer Patients Using Medical Images with Deep Learning

This project proposes a deep learning-based pipeline for predicting survival outcomes of lung cancer patients, by integrating medical imaging (CT scans) with clinical data. It was developed as a capstone project at Ho Chi Minh City University of Technology.

---

## 📌 Project Overview

Lung cancer remains one of the leading causes of death worldwide. Accurate survival prediction is crucial for guiding treatment strategies and improving patient outcomes. This project introduces a two-stage framework:

1. **Lung Tumor Segmentation**: Using the nnU-Net architecture to automatically segment lung tumors from CT images.
2. **Survival Prediction**: Leveraging both extracted radiomic features and clinical data to predict patient survival using traditional statistical models and deep learning methods.

---

## 🧠 Methodology

### 1. Tumor Segmentation
- Utilizes the self-configuring **nnU-Net** framework for 3D CT image segmentation.
- Outputs high-quality tumor masks used for downstream feature extraction.

### 2. Feature Extraction
Two methods are explored:
- **Radiomics-based** (handcrafted features): Extracted with `SimpleITK`, including volume, shape, and intensity statistics.
- **Deep Learning-based**: A dual-branch neural network architecture:
  - **Image branch**: ResNet-based CNN extracts features from CT tumor slices.
  - **Clinical branch**: MLP processes clinical variables (age, sex, stage, etc.)
  - Features are fused and passed to a survival model.

### 3. Survival Analysis
- **Models used**:
  - Cox Proportional Hazards (Cox PH)
  - Accelerated Failure Time (AFT)
  - DeepSurv (deep learning variant of Cox PH)
- **Evaluation Metric**: Concordance Index (C-index)

---

## 📁 Dataset

The experiments use a publicly available dataset (e.g., **NSCLC Radiogenomics**) that includes:
- CT images of lung cancer patients
- Annotated tumor masks
- Associated clinical data (demographics, histology, etc.)

> ⚠️ Note: Due to data privacy constraints, raw medical images are not included in this repository.

---

## 📊 Results

| Model                          | C-Index |
|--------------------------------|---------|
| Clinical data only (Cox PH)    | ~0.62   |
| CT radiomic features only      | ~0.66   |
| Combined (CT + Clinical)       | **0.71** |
| Deep Fusion (Image + Clinical) | ~0.69   |

- The combination of CT-derived features and clinical data achieved the best performance.
- End-to-end deep learning fusion showed promising results, although not surpassing the handcrafted radiomics-based pipeline.

---

## 📌 Key Technologies

- Python, PyTorch, SimpleITK, NumPy, Pandas  
- **nnU-Net** for segmentation  
- **scikit-survival** for Cox/AFT modeling  
- **DeepSurv** (custom PyTorch implementation)

---

## 📚 Report & Documentation

See the full [Capstone Report (PDF)](./Capstone Report.pdf) for technical details, methodology, experimental setup, and analysis.

---

## 🙏 Acknowledgments

This project was conducted under the supervision of **MSc. Võ Thanh Hùng**, Faculty of Computer Science and Engineering, HCMUT.

