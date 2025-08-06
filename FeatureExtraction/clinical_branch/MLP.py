import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import umap

class ClinicalDataset(Dataset):
    """Dataset for clinical data"""
    def __init__(self, features, patient_ids):
        self.features = torch.FloatTensor(features)
        self.patient_ids = patient_ids
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.patient_ids[idx]

class ClinicalMLP(nn.Module):
    def __init__(self, input_dim):
        super(ClinicalMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.network(x)

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_filename = f'logs/clinical_mlp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def evaluate_feature_quality(features, patient_ids):
    metrics = {}
    
    pairwise_distances = pdist(features)
    metrics['avg_pairwise_distance'] = np.mean(pairwise_distances)
    metrics['std_pairwise_distance'] = np.std(pairwise_distances)
    
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(5, len(np.unique(patient_ids))), random_state=42).fit(features)
        metrics['silhouette_score'] = silhouette_score(features, kmeans.labels_)
    except Exception as e:
        logging.warning(f"Silhouette score calculation failed: {e}")
        metrics['silhouette_score'] = None
    
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        plt.title('UMAP Visualization of Clinical Features')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.tight_layout()
        plt.savefig('logs/umap_feature_visualization.png')
        plt.close()
        
        metrics['umap_visualization'] = 'logs/umap_feature_visualization.png'
    except Exception as e:
        logging.warning(f"UMAP visualization failed: {e}")
        metrics['umap_visualization'] = None
    
    return metrics

def save_training_log(train_losses, val_losses, log_filename, metrics):
    with open(log_filename + '_summary.txt', 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Training Losses:\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}\n")
        
        f.write("\nFeature Quality Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_features, _ in train_loader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = outputs.norm(p=2).mean()  # Tiny dummy loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_features, _ in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            loss = outputs.norm(p=2).mean()
            total_loss += loss.item()

    return total_loss / len(val_loader)

def extract_features(model, data_loader, device):
    model.eval()
    all_features = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch_features, batch_patient_ids in data_loader:
            batch_features = batch_features.to(device)
            features = model(batch_features)
            all_features.append(features.cpu().numpy())
            all_patient_ids.append(batch_patient_ids)
    
    return np.concatenate(all_features), np.concatenate(all_patient_ids)

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def main():
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        
        log_filename = setup_logging()
        logging.info("Starting clinical MLP training")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        df = pd.read_csv('processed_data.csv')
        logging.info(f"Loaded data with shape: {df.shape}")
        
        patient_ids = df['Case ID'].values
        features_df = df.drop(columns=['Case ID'])
        features = features_df.values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        X_train, X_val, train_ids, val_ids = train_test_split(
            features_scaled, patient_ids, test_size=0.2, random_state=42
        )
        
        train_dataset = ClinicalDataset(X_train, train_ids)
        val_dataset = ClinicalDataset(X_val, val_ids)
        
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        full_dataset = ClinicalDataset(features_scaled, patient_ids)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = features.shape[1]
        print(f"Input dimension: {input_dim}")
        model = ClinicalMLP(input_dim).to(device)
        logging.info(f"Model Architecture:\n{model}")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        num_epochs = 50
        early_stopping_patience = 50
        best_val_loss = float('inf')
        counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss = validate(model, val_loader, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")    
                    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_clinical_mlp_origin.pth')
                logging.info("Best model saved!")
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    logging.warning(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        plot_training_history(train_losses, val_losses)
        
        model.load_state_dict(torch.load('best_clinical_mlp_origin.pth'))
        
        features_vector, patient_ids_array = extract_features(model, full_loader, device)
        logging.info(f"Extracted features shape: {features_vector.shape}")
        
        np.savez('clinical_features_origin.npz', features=features_vector, patient_ids=patient_ids_array)
        logging.info("Features saved to clinical_features_origin.npz")
        
        feature_metrics = evaluate_feature_quality(features_vector, patient_ids_array)
        
        for metric, value in feature_metrics.items():
            logging.info(f"Feature Metric - {metric}: {value}")
        
        save_training_log(train_losses, val_losses, log_filename, feature_metrics)
        
        logging.info("Training and feature extraction completed successfully!")
    
    except Exception as e:
        logging.critical(f"Main process failed: {e}")
        raise

if __name__ == "__main__":
    main()
