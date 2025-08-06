import numpy as np

# Load file .npz
data = np.load("../../FeatureExtraction/image_branch/method_2/percentile/data_percentile/lung_001.npy")

print(data.shape)
features = data["features"]

print(features)
print("Features shape:", features.shape)  # (115, 512)

# first_sample = features[0]  
# print("First sample shape:", first_sample.shape)  # (512,)
# print("First sample values:", first_sample)  # In toàn bộ 512 đặc trưng của mẫu đầu tiên

# sample_10 = features[10]
# print("Sample 10 values:", sample_10)
