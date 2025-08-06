import SimpleITK as sitk

# Đường dẫn tới folder chứa các file DICOM
dicom_folder = "E:/DATN_LVTN/dataset/NSCLC_Radiogenomics/manifest-1732266091299/NSCLC Radiogenomics/R01-116/03-22-1995-NA-CT THORAX WDYE-99090/1000.000000-ePAD Generated DSO Nov-14-1739-88542"

# Đọc toàn bộ các file DICOM trong folder
reader = sitk.ImageSeriesReader()
dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
reader.SetFileNames(dicom_files)

# Ghép các lát cắt thành ảnh 3D
image = reader.Execute()

# Lấy thông tin Dimensions, Spacing, và Origin
dimensions = image.GetSize()  # Kích thước ảnh (X, Y, Z)
spacing = image.GetSpacing()  # Pixel Spacing và Slice Thickness
origin = image.GetOrigin()    # Tọa độ gốc (Origin)

# In ra các thông số
print(f"Dimensions: {dimensions}")
print(f"Spacing: {spacing}")
print(f"Origin: {origin}")
