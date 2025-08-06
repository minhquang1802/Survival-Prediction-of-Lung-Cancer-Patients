import numpy as np

# Load the .npz file
data = np.load('./clinical_features.npz')

for key in data.files:
    print(f"{key}: {data[key].shape}")