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
from scipy.spatial.distance import pdist, squareform
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
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super(ClinicalMLP, self).__init__()
        
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        
        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded, encoded

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate unique log filename with timestamp
    log_filename = f'logs/clinical_mlp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure logging
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
    """
    Evaluate quality of extracted features using multiple metrics
    
    Args:
        features (np.ndarray): Extracted feature vectors
        patient_ids (np.ndarray): Corresponding patient IDs
    
    Returns:
        dict: Dictionary of feature quality metrics
    """
    metrics = {}
    
    # Pairwise distance distribution
    pairwise_distances = pdist(features)
    metrics['avg_pairwise_distance'] = np.mean(pairwise_distances)
    metrics['std_pairwise_distance'] = np.std(pairwise_distances)
    
    # Silhouette Score (requires labels, using a simple clustering)
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(5, len(np.unique(patient_ids))), random_state=42).fit(features)
        metrics['silhouette_score'] = silhouette_score(features, kmeans.labels_)
    except Exception as e:
        logging.warning(f"Silhouette score calculation failed: {e}")
        metrics['silhouette_score'] = None
    
    # UMAP visualization for dimensionality reduction
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
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
    """
    Save comprehensive training log with losses and feature metrics
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        log_filename (str): Filename of the log
        metrics (dict): Feature quality metrics
    """
    with open(log_filename + '_summary.txt', 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Training Losses:\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}\n")
        
        f.write("\nFeature Quality Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_features, _ in train_loader:
        batch_features = batch_features.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, _ = model(batch_features)
        
        # Compute loss
        loss = criterion(reconstructed, batch_features)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_features, _ in val_loader:
            batch_features = batch_features.to(device)
            
            # Forward pass
            reconstructed, _ = model(batch_features)
            
            # Compute loss
            loss = criterion(reconstructed, batch_features)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def extract_features(model, data_loader, device):
    model.eval()
    all_features = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch_features, batch_patient_ids = batch  # Unpack tuple correctly
            batch_features = batch_features.to(device)
            
            # Extract bottleneck features
            _, bottleneck_features = model(batch_features)
            
            all_features.append(bottleneck_features.cpu().numpy())
            all_patient_ids.append(batch_patient_ids)  # Collect as-is
    
    return np.concatenate(all_features), np.concatenate(all_patient_ids)

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
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
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set up logging
        log_filename = setup_logging()
        logging.info("Starting clinical MLP training")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load clinical data
        df = pd.read_csv('processed_data.csv')
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # Extract patient IDs (assuming row indices as patient IDs if not explicitly given)
        patient_ids = df['Case ID'].values
        
        # Convert data to numpy array
        features_df = df.drop(columns=['Case ID'])
        features = features_df.values
        logging.info(f"Feature shape: {features.shape}")
        
        # Normalize the data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Split data into training and validation sets
        X_train, X_val, train_ids, val_ids = train_test_split(
            features_scaled, patient_ids, test_size=0.2, random_state=42
        )
        
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Validation set shape: {X_val.shape}")
        
        # Create datasets and dataloaders
        train_dataset = ClinicalDataset(X_train, train_ids)
        val_dataset = ClinicalDataset(X_val, val_ids)
        
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create full dataset for final feature extraction
        full_dataset = ClinicalDataset(features_scaled, patient_ids)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = features.shape[1]  # Number of clinical features (45)
        hidden_dims = [256, 512, 1024]  # Increasing dimensions to reach 512 output
        bottleneck_dim = 512  # Matches the required feature vector dimension
        
        model = ClinicalMLP(input_dim, hidden_dims, bottleneck_dim).to(device)
        logging.info(f"Model Architecture:\n{model}")
        
        # Initialize loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training parameters
        num_epochs = 100
        early_stopping_patience = 15
        best_val_loss = float('inf')
        counter = 0
        
        # Track loss history
        train_losses = []
        val_losses = []
        
        try:
            # Training loop
            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = validate(model, val_loader, criterion, device)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")    
                        
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), 'best_clinical_mlp.pth')
                    logging.info("Best model saved!")
                else:
                    counter += 1
                    if counter >= early_stopping_patience:
                        logging.warning(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Plot training history
            plot_training_history(train_losses, val_losses)
            
            # Load best model for feature extraction
            model.load_state_dict(torch.load('best_clinical_mlp.pth'))
            
            # Extract features for all patients
            features_vector, patient_ids_array = extract_features(model, full_loader, device)
            logging.info(f"Extracted features shape: {features_vector.shape}")
            
            # Save features to npz file
            np.savez('clinical_features.npz', features=features_vector, patient_ids=patient_ids_array)
            logging.info("Features saved to clinical_features.npz")
            
            # After feature extraction, evaluate feature quality
            feature_metrics = evaluate_feature_quality(features_vector, patient_ids_array)
            
            # Log feature quality metrics
            for metric, value in feature_metrics.items():
                logging.info(f"Feature Metric - {metric}: {value}")
            
            # Save training log with losses and metrics
            save_training_log(train_losses, val_losses, log_filename, feature_metrics)
            
            logging.info("Training and feature extraction completed successfully!")
        
        except Exception as training_error:
            logging.error(f"Training failed: {training_error}")
            raise
    
    except Exception as main_error:
        logging.critical(f"Main process failed: {main_error}")
        raise

if __name__ == "__main__":
    main()