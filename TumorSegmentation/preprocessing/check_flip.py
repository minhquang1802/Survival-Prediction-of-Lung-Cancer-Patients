import os
import numpy as np
import nibabel as nib
from glob import glob

def dice_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = gt.sum() + pred.sum()
    return 2.0 * intersection / union if union > 0 else 1.0

def check_flipping(gt_file, pred_file):
    gt_img = nib.load(gt_file)
    pred_img = nib.load(pred_file)

    gt = gt_img.get_fdata().astype(np.uint8)
    pred = pred_img.get_fdata().astype(np.uint8)

    if gt.shape != pred.shape:
        print(f"[Shape mismatch] {os.path.basename(gt_file)} - Skipped")
        return

    original_dice = dice_score(gt, pred)
    flip_dice_scores = {}

    for axis, name in enumerate(["X", "Y", "Z"]):
        flipped_pred = np.flip(pred, axis=axis)
        dice = dice_score(gt, flipped_pred)
        flip_dice_scores[name] = dice

    print(f"\n{os.path.basename(gt_file)}:")
    print(f"Original Dice: {original_dice:.4f}")
    for axis_name, dice in flip_dice_scores.items():
        print(f"Flip {axis_name}-axis Dice: {dice:.4f}")

    # Gợi ý chiều bị lật nếu Dice sau flip tăng rõ rệt
    for axis_name, dice in flip_dice_scores.items():
        if dice - original_dice > 0.2:  # Ngưỡng có thể điều chỉnh
            print(f"Có thể bị lật theo trục {axis_name}")

def run_check(gt_folder, pred_folder):
    gt_files = sorted(glob(os.path.join(gt_folder, "*.nii.gz")))
    pred_files = sorted(glob(os.path.join(pred_folder, "*_seg.nii.gz")))

    gt_dict = {os.path.basename(f).replace("_0000.nii.gz", ".nii.gz"): f for f in gt_files}

    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path).replace("_0000_seg.nii.gz", ".nii.gz")
        gt_path = gt_dict.get(pred_name)
        if gt_path and os.path.exists(gt_path):
            check_flipping(gt_path, pred_path)
        else:
            print(f"Không tìm thấy ground truth tương ứng với {pred_name}")

# === Sử dụng ===
if __name__ == "__main__":
    ground_truth_folder = "E:/DATN_LVTN/TumorSegmentation/train/nnUNet_raw/Dataset015_lungTumor/labelsTs"   # ví dụ: nnUNet_raw/DatasetXXX_lungTumor/labelsTs
    prediction_folder  = "E:/DATN_LVTN/TumorSegmentation/UNet_2/inference"   # ví dụ: UNet_2/inference
    run_check(ground_truth_folder, prediction_folder)
