import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression

# Đọc dữ liệu
df = pd.read_csv('processed_data.csv')

# 3. Chuyển đổi biến phân loại thành biến giả (dummy variables)
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col not in ['time_to_event', 'event']])

# 4. Chuẩn bị dữ liệu cho mô hình
# Giả sử cột thời gian được gọi là 'time_to_event' và cột sự kiện là 'event'
time_col = 'Time to Event'
event_col = 'Event'

# Chia tất cả các biến còn lại làm đặc trưng
feature_cols = [col for col in df.columns if col not in [time_col, event_col]]

X = df[feature_cols]
y_time = df[time_col]
y_event = df[event_col]

# 5. Phân tích mô tả
print("\nThống kê mô tả:")
print(df.describe())

# 6. Biểu đồ Kaplan-Meier cơ bản (để so sánh với Weibull)
plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()
kmf.fit(y_time, y_event, label='Kaplan-Meier Estimate')
kmf.plot()
plt.title('Đường cong Kaplan-Meier')
plt.ylabel('Xác suất sống sót')
plt.xlabel('Thời gian')
add_at_risk_counts(kmf)
plt.grid(True)
plt.tight_layout()
plt.savefig('kaplan_meier_curve.png')
plt.close()

# 7. Mô hình Weibull cơ bản (không có covariates)
plt.figure(figsize=(10, 6))
wbf = WeibullFitter()
wbf.fit(y_time, y_event, label='Weibull Model')
wbf.plot()

# So sánh với Kaplan-Meier
kmf.plot(ax=plt.gca())
plt.title('So sánh mô hình Weibull với Kaplan-Meier')
plt.ylabel('Xác suất sống sót')
plt.xlabel('Thời gian')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('weibull_vs_km.png')
plt.close()

# In các tham số của mô hình Weibull
print("\nCác tham số của mô hình Weibull:")
print(wbf.summary)

# 8. Phân tích với các covariates
# Chọn các đặc trưng quan trọng nhất
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X, y_time)
selected_features = X.columns[selector.get_support()]
print("\nCác đặc trưng quan trọng nhất:", selected_features.tolist())

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 9. Phân tích ảnh hưởng của các đặc trưng chính đến thời gian sống còn
from lifelines import CoxPHFitter

# Tạo DataFrame mới với đặc trưng đã chọn, thời gian và sự kiện
df_cox = pd.concat([X_scaled_df[selected_features], 
                    pd.DataFrame({time_col: y_time, event_col: y_event})], axis=1)

cph = CoxPHFitter()
cph.fit(df_cox, duration_col=time_col, event_col=event_col)
print("\nKết quả mô hình Cox PH:")
print(cph.summary)

# Vẽ biểu đồ ảnh hưởng của các đặc trưng
plt.figure(figsize=(10, 6))
cph.plot()
plt.title('Ảnh hưởng của các đặc trưng đến nguy cơ')
plt.tight_layout()
plt.savefig('cox_features_impact.png')
plt.close()

# 10. Ước tính thời gian sống còn trung vị
median_survival_time = wbf.median_survival_time_
print(f"\nThời gian sống còn trung vị (Weibull): {median_survival_time:.2f}")
print(f"Thời gian sống còn trung vị (Kaplan-Meier): {kmf.median_survival_time_:.2f}")

# 11. Dự đoán xác suất sống sót ở các mốc thời gian khác nhau
time_points = [1, 3, 5, 10]  # Điều chỉnh theo đơn vị thời gian của bạn
print("\nXác suất sống sót theo mô hình Weibull:")
for t in time_points:
    prob = wbf.predict(t)
    print(f"Tại thời điểm {t}: {prob:.4f}")

# 12. Chia tập dữ liệu để đánh giá mô hình
X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
    X, y_time, y_event, test_size=0.3, random_state=42
)

# 13. Đánh giá mô hình trên tập kiểm tra
wbf_test = WeibullFitter()
wbf_test.fit(time_train, event_train)

# Tính chỉ số concordance sử dụng hàm dự đoán thời gian sống còn trung vị
# Sửa lỗi: Thay predict_median bằng cách sử dụng median_survival_time_
predicted_times = np.ones(len(time_test)) * wbf_test.median_survival_time_
c_index = concordance_index(time_test, -predicted_times, event_test)  # Dấu - vì giá trị thấp hơn tương ứng với nguy cơ cao hơn
print(f"\nChỉ số concordance: {c_index:.4f}")

# 14. Phân tích tham số Weibull
print("\nThông số phân phối Weibull:")
print(f"Tham số hình dạng (shape): {wbf.rho_:.4f}")
print(f"Tham số tỷ lệ (scale): {wbf.lambda_:.4f}")

# Diễn giải kết quả
if wbf.rho_ > 1:
    print("Tham số hình dạng > 1: Nguy cơ tăng theo thời gian")
elif wbf.rho_ < 1:
    print("Tham số hình dạng < 1: Nguy cơ giảm theo thời gian")
else:
    print("Tham số hình dạng ≈ 1: Nguy cơ không đổi theo thời gian (gần giống phân phối Exponential)")

# 15. Phân tích Weibull AFT (Accelerated Failure Time)
from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter()
df_aft = pd.concat([X_scaled_df, pd.DataFrame({time_col: y_time, event_col: y_event})], axis=1)
aft.fit(df_aft, duration_col=time_col, event_col=event_col)

print("\nKết quả mô hình Weibull AFT:")
print(aft.summary)

# Vẽ biểu đồ ảnh hưởng của các đặc trưng
plt.figure(figsize=(12, 8))
aft.plot()
plt.title('Ảnh hưởng của các đặc trưng trong mô hình Weibull AFT')
plt.tight_layout()
plt.savefig('weibull_aft_features.png')
plt.close()

# 16. Phân tích và trực quan hóa hàm nguy cơ (hazard function)
plt.figure(figsize=(10, 6))
t = np.linspace(0, max(y_time), 100)
hazard = wbf.hazard_at_times(t)
plt.plot(t, hazard)
plt.title('Hàm nguy cơ (hazard function) theo mô hình Weibull')
plt.xlabel('Thời gian')
plt.ylabel('Tỷ lệ nguy cơ')
plt.grid(True)
plt.tight_layout()
plt.savefig('weibull_hazard_function.png')
plt.close()

# 17. Tóm tắt kết quả
print("\n======= TÓM TẮT KẾT QUẢ PHÂN TÍCH SỐNG CÒN =======")
print(f"Số mẫu: {len(df)}")
print(f"Số sự kiện xảy ra: {sum(y_event)}")
print(f"Tỷ lệ sự kiện: {sum(y_event)/len(df):.2%}")
print(f"Thời gian sống còn trung vị: {median_survival_time:.2f}")
print(f"Xác suất sống sót sau 5 đơn vị thời gian: {wbf.predict(5):.4f}")
print(f"Chỉ số concordance: {c_index:.4f}")
print("=================================================")

# 18. Bổ sung: Vẽ đường cong sống còn dự đoán cho các giá trị đặc trưng khác nhau
# Giả sử chúng ta muốn xem xét ảnh hưởng của một đặc trưng quan trọng nhất
if len(selected_features) > 0:
    important_feature = selected_features[0]
    plt.figure(figsize=(10, 6))
    
    # Tạo hai bộ dữ liệu, một cho giá trị cao và một cho giá trị thấp của đặc trưng
    df_high = df_aft.copy()
    df_low = df_aft.copy()
    
    # Đặt giá trị đặc trưng quan trọng nhất là cao (1 độ lệch chuẩn) và thấp (-1 độ lệch chuẩn)
    df_high[important_feature] = 1.0
    df_low[important_feature] = -1.0
    
    # Dự đoán đường cong sống còn
    aft.plot_partial_effects_on_outcome(important_feature, values=[df_low[important_feature].iloc[0], df_high[important_feature].iloc[0]], 
                                         cmap='coolwarm', ax=plt.gca())
    plt.title(f'Ảnh hưởng của {important_feature} đến xác suất sống còn')
    plt.ylabel('Xác suất sống sót')
    plt.xlabel('Thời gian')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('feature_impact_survival.png')
    plt.close()