import nibabel as nib

img = nib.load("example.nii.gz")
data = img.get_fdata()        # ndarray chứa voxel data
affine = img.affine           # ma trận affine (biến đổi tọa độ)
header = img.header           # metadata
