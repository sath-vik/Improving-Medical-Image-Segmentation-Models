import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target):
    smooth = 1e-5

    # Convert to numpy and threshold
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_bin = output > 0.5
    target_bin = target > 0.5 # Ensure target is also binary

    # Handle potential batch dimension by calculating intersection/union per sample
    axis_to_sum = tuple(range(1, output_bin.ndim)) # Sum over H, W (, D) axes
    intersection = (output_bin & target_bin).sum(axis=axis_to_sum)
    union = (output_bin | target_bin).sum(axis=axis_to_sum)

    # Calculate IoU per sample, handle union=0 case
    iou_per_sample = np.where(
        union == 0,
        1.0, # Both empty, IoU is 1
        (intersection + smooth) / (union + smooth)
    )

    # Calculate Dice per sample, handle sums=0 case
    output_sum = output_bin.sum(axis=axis_to_sum)
    target_sum = target_bin.sum(axis=axis_to_sum)
    dice_per_sample = np.where(
        (output_sum + target_sum) == 0,
        1.0, # Both empty, Dice is 1
        (2. * intersection + smooth) / (output_sum + target_sum + smooth)
    )

    # Average over batch
    iou = np.mean(iou_per_sample)
    dice = np.mean(dice_per_sample)

    # Removed hd95 from here - calculate it in indicators
    return iou, dice, False # last value seems unused

def indicators(output, target):
    # Ensure tensors are on CPU and detached
    output_detached = output.detach().cpu()
    target_detached = target.detach().cpu()

    # Sigmoid and thresholding
    output_sig = torch.sigmoid(output_detached)
    output_bin = output_sig > 0.5 # Binary prediction mask (True/False)

    # Convert to NumPy uint8 arrays
    # Handle batch and channel dimension
    # Prediction (B, C, H, W) -> Assume C=1 for binary seg -> (B, H, W)
    if output_bin.ndim == 4 and output_bin.shape[1] == 1:
        output_np = output_bin[:, 0, :, :].numpy().astype(np.uint8)
    elif output_bin.ndim == 3: # Already (B, H, W)
        output_np = output_bin.numpy().astype(np.uint8)
    else:
        raise ValueError(f"Unexpected prediction dimensions: {output_bin.shape}")

    # Target (B, C, H, W) -> Assume C=1 -> (B, H, W)
    if target_detached.ndim == 4 and target_detached.shape[1] == 1:
         target_np = target_detached[:, 0, :, :].numpy().astype(np.uint8)
    elif target_detached.ndim == 3: # Already (B, H, W)
         target_np = target_detached.numpy().astype(np.uint8)
    else:
        raise ValueError(f"Unexpected target dimensions: {target_detached.shape}")

    # --- Initialize metrics lists ---
    batch_size = output_np.shape[0]
    iou_vals = []
    dice_vals = []
    hd_vals = []
    hd95_vals = []
    recall_vals = []
    specificity_vals = []
    precision_vals = []

    # --- Iterate through batch ---
    for i in range(batch_size):
        pred_mask = output_np[i] # Single prediction mask HxW
        gt_mask = target_np[i]   # Single ground truth mask HxW

        # Calculate metrics using medpy for EACH sample
        # Medpy requires inputs to be bool or int-like
        iou_ = jc(pred_mask, gt_mask)
        dice_ = dc(pred_mask, gt_mask)

        # --- Check for empty masks before HD/HD95 ---
        hd_ = np.nan
        hd95_ = np.nan
        pred_empty = not np.any(pred_mask)
        gt_empty = not np.any(gt_mask)

        if not pred_empty and not gt_empty:
            # Both masks have foreground pixels, calculate HD/HD95
            try:
                # Voxel spacing might be relevant (usually [1,1] for 2D)
                hd_ = hd(pred_mask, gt_mask, voxelspacing=[1,1])
                hd95_ = hd95(pred_mask, gt_mask, voxelspacing=[1,1])
            except RuntimeError as e:
                # Catch rare errors from medpy even with checks
                print(f"Warning: medpy HD calculation failed for sample {i}: {e}")
                # Keep hd_, hd95_ as np.nan
        elif gt_empty and pred_empty:
             # Both empty: HD/HD95 is 0
             hd_ = 0.0
             hd95_ = 0.0
        # Else: one is empty -> HD/HD95 are undefined -> keep as np.nan
        # -----------------------------------------

        # Recall, Precision, Specificity using medpy (handle potential division by zero)
        # Medpy functions often return NaN if denominator is zero, which is fine
        recall_ = recall(pred_mask, gt_mask)
        specificity_ = specificity(pred_mask, gt_mask)
        precision_ = precision(pred_mask, gt_mask)

        # Append sample metrics to lists
        iou_vals.append(iou_)
        dice_vals.append(dice_)
        hd_vals.append(hd_)
        hd95_vals.append(hd95_)
        recall_vals.append(recall_)
        specificity_vals.append(specificity_)
        precision_vals.append(precision_)

    # --- Average metrics over the batch (handle NaNs) ---
    final_iou = np.nanmean(iou_vals) if len(iou_vals) > 0 else np.nan
    final_dice = np.nanmean(dice_vals) if len(dice_vals) > 0 else np.nan
    final_hd = np.nanmean(hd_vals) if len(hd_vals) > 0 else np.nan
    final_hd95 = np.nanmean(hd95_vals) if len(hd95_vals) > 0 else np.nan
    final_recall = np.nanmean(recall_vals) if len(recall_vals) > 0 else np.nan
    final_specificity = np.nanmean(specificity_vals) if len(specificity_vals) > 0 else np.nan
    final_precision = np.nanmean(precision_vals) if len(precision_vals) > 0 else np.nan

    # Return average metrics for the batch
    return final_iou, final_dice, final_hd, final_hd95, final_recall, final_specificity, final_precision
