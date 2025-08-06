import numpy as np
import torch
import torch.nn as nn

def load_features(image_path, clinical_path):
    """
    Load feature vectors from NPZ files
    
    Args:
        image_path (str): Path to image branch feature vector file
        clinical_path (str): Path to clinical branch feature vector file
    
    Returns:
        tuple: (image_features, clinical_features, patient_ids)
    """
    image_data = np.load(image_path)
    clinical_data = np.load(clinical_path)
    
    image_features = image_data['features']
    clinical_features = clinical_data['features']
    patient_ids = image_data['patient_ids']
    
    return image_features, clinical_features, patient_ids

class FeatureFusionLayer(nn.Module):
    def __init__(self, image_feature_dim=512, clinical_feature_dim=512, fusion_dim=1024):
        """
        Feature fusion layer with concatenation and optional projection
        
        Args:
            image_feature_dim (int): Dimension of image branch features
            clinical_feature_dim (int): Dimension of clinical branch features
            fusion_dim (int): Dimension of fused feature vector
        """
        super(FeatureFusionLayer, self).__init__()
        
        # Simple concatenation layer
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + clinical_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, image_features, clinical_features):
        """
        Concatenate and project features
        
        Args:
            image_features (torch.Tensor): Features from image branch
            clinical_features (torch.Tensor): Features from clinical branch
        
        Returns:
            torch.Tensor: Fused feature vector
        """
        # Concatenate features
        fused_features = torch.cat([image_features, clinical_features], dim=1)
        
        # Project to new feature space
        output = self.fusion(fused_features)
        
        return output

# Example usage
def main():
    # Paths to your NPZ files
    image_path = '../FeatureExtraction/image_branch/method_2/mean/result/features_attention.npz'
    clinical_path = '../FeatureExtraction/clinical_branch/clinical_features_origin.npz'

    output_path = '../FeatureExtraction/fusion_features_origin.npz'

    
    # Load features
    image_features, clinical_features, patient_ids = load_features(image_path, clinical_path)
    
    # Convert to PyTorch tensors
    image_tensor = torch.tensor(image_features, dtype=torch.float32)
    clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32)
    
    # Initialize fusion layer
    fusion_layer = FeatureFusionLayer()
    
    # Perform feature fusion
    fused_features = fusion_layer(image_tensor, clinical_tensor)
    fused_features_np = fused_features.detach().numpy()
    
    # Save fused features to NPZ file
    np.savez(
        output_path, 
        features=fused_features_np,  # Lưu feature vector sau fusion
        patient_ids=patient_ids      # Giữ nguyên patient_ids
    )
    
    print("Fused features shape:", fused_features.shape)
    print(f"Fused features saved to {output_path}")

if __name__ == "__main__":
    main()