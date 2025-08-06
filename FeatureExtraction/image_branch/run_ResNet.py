import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    model = models.resnet34(pretrained=True)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Remove the last fully connected layer to get feature vector
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def get_transforms():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

class SliceDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = np.load(file_path)
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[2]
    
    def __getitem__(self, idx):
        slice_data = self.data[:, :, idx, :]
        slice_data = (slice_data * 255).astype(np.uint8)
        
        if self.transform:
            slice_data = self.transform(slice_data)
            
        return slice_data

def extract_features_from_file(file_path, feature_extractor, transform, batch_size=16):
    dataset = SliceDataset(file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            features = feature_extractor(batch)
            features = features.reshape(features.size(0), -1).cpu().numpy()
            all_features.append(features)
    
    all_features = np.concatenate(all_features, axis=0)
    return all_features  # Shape: [num_slices, 512]

def aggregate_features(features, method='mean'):
    if method == 'mean':
        return np.mean(features, axis=0)
    elif method == 'max':
        return np.max(features, axis=0)
    elif method == 'attention':
        attention_scores = np.linalg.norm(features, axis=1)
        exp_scores = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        weighted_features = features * attention_weights[:, np.newaxis]
        return np.sum(weighted_features, axis=0)
    else:
        raise ValueError(f"Aggregation method {method} is not supported.")

def process_dataset(data_dir, output_dir, split='train', agg_method='attention'):
    os.makedirs(output_dir, exist_ok=True)
    
    feature_extractor = get_feature_extractor()
    transform = get_transforms()
    
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    patient_features = {}
    patient_ids = []
    
    for npy_file in tqdm(npy_files, desc=f"Processing {split} files"):
        file_path = os.path.join(data_dir, npy_file)
        patient_id = npy_file.split('.')[0]
        patient_ids.append(patient_id)
        
        slice_features = extract_features_from_file(file_path, feature_extractor, transform)
        
        patient_feature = aggregate_features(slice_features, method=agg_method)
        
        patient_features[patient_id] = patient_feature
    
    output_file = os.path.join(output_dir, f"{split}_features_{agg_method}.npz")
    np.savez(output_file, 
             features=np.array([patient_features[pid] for pid in patient_ids]),
             patient_ids=patient_ids)
    print(f"Feature vectors saved to {output_file}")
    
    import h5py
    h5_output_file = os.path.join(output_dir, f"{split}_features_{agg_method}.h5")
    with h5py.File(h5_output_file, 'w') as f:
        feature_group = f.create_group('resnet_features')
        for pid, feature in patient_features.items():
            feature_group.create_dataset(pid, data=feature)
    
    return patient_features, patient_ids

def evaluate_features(patient_features, patient_ids, output_dir, split='train'):
    feature_array = np.array([patient_features[pid] for pid in patient_ids])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_array)
    
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_array)-1))
    features_tsne = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.7)
    
    for i, pid in enumerate(patient_ids):
        plt.annotate(pid, (features_tsne[i, 0], features_tsne[i, 1]), fontsize=8)
        
    plt.title(f't-SNE Visualization of {split} Features')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split}_tsne.png"), dpi=300)
    plt.close()

    print("Computing Silhouette Score...")
    from sklearn.cluster import KMeans
    
    silhouette_scores = []
    max_clusters = min(10, len(feature_array) - 1)
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        try:
            score = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(score)
            print(f"  Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}")
        except:
            print(f"  Unable to compute Silhouette Score for {n_clusters} clusters")
            silhouette_scores.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(cluster_range), silhouette_scores, 'o-')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{split}_silhouette_scores.png"), dpi=300)
    plt.close()
    
    from sklearn.decomposition import PCA
    print("Computing PCA Variance Explained...")
    
    pca = PCA().fit(features_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"  Number of components to explain 95% variance: {n_components_95}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-')
    plt.title('Individual Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'o-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance explained')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split}_pca_variance.png"), dpi=300)
    plt.close()
    
    metrics_results = {
        'pca_n_components_95': int(n_components_95),
        'silhouette_scores': {int(n): float(score) for n, score in zip(cluster_range, silhouette_scores)},
        'best_n_clusters': int(list(cluster_range)[np.argmax(silhouette_scores)]),
        'best_silhouette_score': float(max(silhouette_scores)) if silhouette_scores else None
    }
    
    import json
    with open(os.path.join(output_dir, f"{split}_metrics.json"), 'w') as f:
        json.dump(metrics_results, f, indent=4)
    
    return metrics_results

if __name__ == "__main__":
    input_dir = "../../FeatureExtraction/image_branch/method_2/percentile/data_percentile"
    output_dir = "../../FeatureExtraction/image_branch/method_2/percentile/result_percentile"
    
    aggregation_method = 'attention'
    
    print("Processing data...")
    train_features, train_ids = process_dataset(input_dir, output_dir, 'train', aggregation_method)
    print(f"Completed processing {len(train_features)} data files.")
    
    print("\nEvaluating data feature vectors...")
    train_metrics = evaluate_features(train_features, train_ids, output_dir, 'train')
    
    print("\nData evaluation results:")
    print(f"- Number of PCA components needed to retain 95% variance: {train_metrics['pca_n_components_95']}")
    print(f"- Best number of clusters according to Silhouette score: {train_metrics['best_n_clusters']}")
    print(f"- Best Silhouette score: {train_metrics['best_silhouette_score']:.4f}")