import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("processed_data.csv")

# Select numerical features to normalize
columns_to_scale = [
    "Volume", "Mean", "Std", "Min", "Max", "Median", 
    "SurfaceArea", "Elongation", "Flatness", "Roundness"
]

# Apply Z-score normalization
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Save the normalized dataset
df.to_csv("processed_data_zscore.csv", index=False)

# Display the first few rows of the normalized dataset
print(df.head())