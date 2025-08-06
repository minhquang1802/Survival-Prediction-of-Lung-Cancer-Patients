import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

def prepare_survival_data(fusion_features_path, clinical_csv_path):
    """
    Chuẩn bị dữ liệu sinh tồn từ feature vector và file CSV
    """
    # Nạp feature vector đã fusion
    fusion_data = np.load(fusion_features_path)
    fusion_patient_ids = fusion_data['patient_ids']
    features = fusion_data['features']
    
    # Nạp dữ liệu lâm sàng
    clinical_df = pd.read_csv(clinical_csv_path)
    
    # Khởi tạo danh sách để lưu thời gian và nhãn sự kiện
    survival_times = []
    event_labels = []
    
    # Khớp patient IDs và trích xuất thông tin
    for patient_id in fusion_patient_ids:
        # Tìm hàng tương ứng với patient ID
        patient_row = clinical_df[clinical_df['Case ID'] == patient_id]
        
        if len(patient_row) == 0:
            raise ValueError(f"Không tìm thấy patient ID {patient_id} trong dữ liệu lâm sàng")
        
        # Trích xuất thời gian sống sót và nhãn sự kiện
        survival_times.append(patient_row['Time to Event'].values[0])
        event_labels.append(patient_row['Survival Status'].values[0])
    
    # Chuyển sang NumPy array
    survival_times = np.array(survival_times)
    event_labels = np.array(event_labels)
    
    return features, survival_times, event_labels

def train_pycox_deepsurv(features, survival_times, event_labels, test_size=0.2):
    """
    Huấn luyện mô hình DeepSurv sử dụng Pycox
    """
    # Chia tập train/val/test
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        features, survival_times, event_labels, 
        test_size=test_size, random_state=42
    )
    
    # Chia tập validation từ training
    X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
        X_train, T_train, E_train, 
        test_size=0.2, random_state=42
    )
    
    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    X_test_df = pd.DataFrame(X_test)
    
    scaler = StandardScaler()
    
    # Chuyển đổi dữ liệu
    X_train = scaler.fit_transform(X_train_df).astype('float32')
    X_val = scaler.transform(X_val_df).astype('float32')
    X_test = scaler.transform(X_test_df).astype('float32')
    
    # Chuẩn bị validation data
    val = (X_val, (T_val, E_val))
    
    # Định nghĩa mạng neural
    in_features = X_train.shape[1]
    net = tt.practical.MLPVanilla(
        in_features, 
        [64, 32],  # Các layers ẩn
        out_features=1, 
        dropout=0.3
    )
    
    # Khởi tạo mô hình CoxPH
    model = CoxPH(net, tt.optim.Adam)
    
    # Hyperparameters
    batch_size = 32
    epochs = 100
    
    # Callbacks
    callbacks = [
        tt.callbacks.EarlyStopping(patience=10)
    ]
    
    # Huấn luyện mô hình
    log = model.fit(
        X_train, 
        (T_train, E_train),  # Tuple (thời gian, sự kiện)
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=True,
        val_data=val,
        val_batch_size=batch_size
    )
    
    # Đánh giá mô hình
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(X_test)
    ev = EvalSurv(surv, T_test, E_test, censor_surv='km')
    concordance_index = ev.concordance_td()
    
    return {
        'model': model,
        'concordance_index': concordance_index,
        'survival_curves': surv,
        'mapper': scaler
    }

def main():
    # Đường dẫn file
    fusion_features_path = 'fusion_features.npz'
    clinical_csv_path = 'processed_data.csv'
    
    try:
        # Chuẩn bị dữ liệu
        features, survival_times, event_labels = prepare_survival_data(
            fusion_features_path, 
            clinical_csv_path
        )
        
        # Huấn luyện mô hình
        result = train_pycox_deepsurv(features, survival_times, event_labels)
        
        # In kết quả
        print("\nKết quả huấn luyện:")
        print(f"Concordance Index: {result['concordance_index']:.4f}")
        
        # Lưu mô hình
        torch.save(result['model'].net.state_dict(), 'pycox_deepsurv_model.pth')
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()