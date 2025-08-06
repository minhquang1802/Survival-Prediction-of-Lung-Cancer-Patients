from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd


# Read image and mask
image = sitk.ReadImage("../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/imagesTr/lung_042_0000.nii.gz")
mask = sitk.ReadImage("../../../TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTr/lung_042.nii.gz")

# Create default extractor, without using wavelet/Gabor (for simplicity)
extractor = featureextractor.RadiomicsFeatureExtractor()

# (Optional) configure to reduce number of features if needed:
extractor.enableAllFeatures()  # default: enable all
# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName("firstorder")
# extractor.enableFeatureClassByName("shape")

# Extract features
result = extractor.execute(image, mask)

# Convert to dataframe for display
df = pd.DataFrame.from_dict(result, orient="index", columns=["Value"])
df = df.sort_index()  # for easier viewing

# Save to file if needed
df.to_csv("radiomics_features_42.csv")
print(df.head(20))
