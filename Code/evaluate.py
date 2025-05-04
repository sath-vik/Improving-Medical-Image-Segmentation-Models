# evaluate.py
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
import argparse
import os
from glob import glob
import numpy as np
from tqdm import tqdm

# Import project modules AFTER conditional logic if they depend on arch_module
# Removed imports that might depend on the architecture choice for now

# Define parse_args FIRST
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for evaluation')
    parser.add_argument('--input_w', default=256, type=int, help='Input image width')
    parser.add_argument('--input_h', default=256, type=int, help='Input image height')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loader workers')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN', help='Model architecture')
    parser.add_argument('--deep_supervision', default=False, type=str2bool, help='Use deep supervision?')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes')
    parser.add_argument('--loss', default='BCEDiceLoss', help='Loss function for evaluation (optional)')
    parser.add_argument('--no_kan', action='store_true', help='Disable KAN layers if applicable')
    # Use a helper function or ensure list type conversion if needed later
    parser.add_argument('--input_list', type=str, default='128,160,256', help='Comma-separated list for embed_dims')
    parser.add_argument('--upgrade', action='store_true',
                    help='Specify if the model being evaluated was trained using the upgraded architectures.')
    parser.add_argument('--dataset', type=str, required=True,
                    choices=['ISIC', 'Lumbar_Spine', 'Fetus_Head'],
                    help='Dataset name for evaluation (e.g., ISIC, Lumbar_Spine)')
    parser.add_argument('--input_channels', default=1, type=int,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    args = parser.parse_args()

    # Convert input_list from string to list of ints
    try:
        args.input_list = [int(i) for i in args.input_list.split(',')]
    except ValueError:
        raise ValueError("Invalid format for --input_list. Use comma-separated integers (e.g., 128,160,256).")

    return args

# --- Define other functions BEFORE main ---
from metrics import iou_score, indicators #Import your metric.py
from utils import AverageMeter, str2bool  # Import if you use AverageMeter
# Import losses here as it's needed in main
import losses

def calculate_metrics(output, target):
    """Calculates IoU, Dice, F1, accuracy, precision, recall, specificity."""
    iou, dice, _ = iou_score(output, target)
    # Use the indicators function to get additional metrics, as defined in your metrics.py
    iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)
    # Calculate accuracy.  Note that iou_score returns *average* IoU and Dice,
    # so we can use those directly.
    f1_score = (2 * precision_ * recall_) / (precision_ + recall_ + 1e-7)  # Add epsilon to prevent division by zero
    # hd and hd95 are also calculated
    return iou, dice, f1_score, precision_, recall_, specificity_, hd_, hd95_

def evaluate(config, model, val_loader, criterion=None, device='cpu'): # Pass config and device
    """Evaluates the model on the validation data and calculates metrics."""
    avg_meters = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'f1': AverageMeter(),
        # 'accuracy': AverageMeter(), # Accuracy might need different calculation depending on definition
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'specificity': AverageMeter(),
        'hd': AverageMeter(),    # Hausdorff Distance
        'hd95': AverageMeter(),  # 95th percentile Hausdorff Distance
    }
    if criterion:
        avg_meters['loss'] = AverageMeter()

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations
        pbar = tqdm(total=len(val_loader), desc="Evaluating")
        for input_data, target_data, _ in val_loader: # Renamed to avoid conflict
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            # Handle potential deep supervision output during evaluation if needed
            if config.deep_supervision:
                 outputs = model(input_data)
                 output = outputs[-1] # Usually evaluate the final output
                 if criterion:
                      loss = 0
                      for o in outputs:
                           loss += criterion(o, target_data)
                      loss /= len(outputs)
                      avg_meters['loss'].update(loss.item(), input_data.size(0))
            else:
                 output = model(input_data)
                 if criterion:
                      loss = criterion(output, target_data)
                      avg_meters['loss'].update(loss.item(), input_data.size(0))


            iou, dice, f1, precision, recall, specificity, hd, hd95 = calculate_metrics(output, target_data) # Use target_data


            avg_meters['iou'].update(iou, input_data.size(0))
            avg_meters['dice'].update(dice, input_data.size(0))
            avg_meters['f1'].update(f1, input_data.size(0))
            avg_meters['precision'].update(precision, input_data.size(0))
            avg_meters['recall'].update(recall, input_data.size(0))
            avg_meters['specificity'].update(specificity, input_data.size(0))
            # Handle potential NaN/inf values from Hausdorff distance if mask is empty
            if not np.isnan(hd) and not np.isinf(hd):
                 avg_meters['hd'].update(hd, input_data.size(0))
            if not np.isnan(hd95) and not np.isinf(hd95):
                 avg_meters['hd95'].update(hd95, input_data.size(0))


            pbar.set_postfix({
                'IoU': f"{avg_meters['iou'].avg:.4f}",
                'Dice': f"{avg_meters['dice'].avg:.4f}",
            })
            pbar.update(1)
        pbar.close()

    # Prepare final results, handling cases where HD might not have been updated
    final_results = OrderedDict()
    if criterion:
        final_results['loss'] = avg_meters['loss'].avg
    final_results['iou'] = avg_meters['iou'].avg
    final_results['dice'] = avg_meters['dice'].avg
    final_results['f1'] = avg_meters['f1'].avg
    final_results['precision'] = avg_meters['precision'].avg
    final_results['recall'] = avg_meters['recall'].avg
    final_results['specificity'] = avg_meters['specificity'].avg
    final_results['hd'] = avg_meters['hd'].avg if avg_meters['hd'].count > 0 else float('nan') # Report NaN if no valid HD calculated
    final_results['hd95'] = avg_meters['hd95'].avg if avg_meters['hd95'].count > 0 else float('nan') # Report NaN if no valid HD95 calculated

    return final_results


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # 1. Parse Arguments ONCE
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Conditional Architecture Import based on Parsed Config
    if config.upgrade:
        try:
            import archs_upgraded as arch_module
            print("INFO: Using UPGRADED architectures from archs_upgraded.py for evaluation")
        except ImportError:
            print("ERROR: --upgrade flag was set, but archs_upgraded.py could not be imported.")
            exit(1)
    else:
        try:
            import archs as arch_module
            print("INFO: Using standard architectures from archs.py for evaluation")
        except ImportError:
            print("ERROR: archs.py could not be imported.")
            exit(1)

    # 3. Import Dataset classes AFTER potential arch imports if they depend on it
    # (Usually safe, but good practice)
    from dataset import Dataset
    from lumbarSpineDataset import LumbarSpineDataset

    # --- 4. Create Dataset and DataLoader ---
    print(f"Loading evaluation data for dataset: {config.dataset}")
    print(f"Data base directory: {config.data_dir}")

    val_transform = A.Compose([
        A.Resize(config.input_h, config.input_w),
        A.Normalize(),
    ])

    # --- Dataset Loading Logic ---
    if config.dataset == 'Lumbar_Spine':
        img_ext = '.png'
        mask_ext = '.png'
        base_dir = os.path.join(config.data_dir, 'png')
        image_dir = os.path.join(base_dir, 'images')
        mask_dir = os.path.join(base_dir, 'masks')
        print(f"Expecting Lumbar images in: {image_dir}")
        print(f"Expecting Lumbar masks in: {mask_dir}")

        all_image_files = []
        if not os.path.isdir(image_dir):
             raise FileNotFoundError(f"Lumbar Spine image directory not found: {image_dir}")
        for case_folder in sorted(os.listdir(image_dir)):
            case_path_img = os.path.join(image_dir, case_folder)
            if os.path.isdir(case_path_img):
                all_image_files.extend(sorted(glob(os.path.join(case_path_img, '*' + img_ext))))

        val_img_ids = []
        val_mask_ids = []
        if not os.path.isdir(mask_dir):
             print(f"Warning: Lumbar Spine mask directory not found: {mask_dir}. Cannot load masks for evaluation.")
        else:
             for img_file in all_image_files:
                  img_filename = os.path.basename(img_file)
                  case_folder = os.path.basename(os.path.dirname(img_file))
                  mask_filename = img_filename.replace('Slice', 'Labels_Slice')
                  mask_path = os.path.join(mask_dir, case_folder, mask_filename)

                  if os.path.exists(mask_path):
                       val_img_ids.append(img_file)
                       val_mask_ids.append(mask_path)
                  # else: # Optional: Be less verbose during evaluation
                  #      print(f"Warning: Mask not found for evaluation image: {img_file}")

        if not val_img_ids:
             raise FileNotFoundError(f"No evaluation images found for Lumbar Spine in {image_dir} with matching masks in {mask_dir}")

        val_dataset = LumbarSpineDataset(
            img_ids=val_img_ids,
            mask_ids=val_mask_ids,
            img_dir="",
            mask_dir="",
            transform=val_transform
        )

    elif config.dataset == 'ISIC':
        img_ext = '.jpg'
        mask_ext = '_segmentation.png'
        test_img_dir = os.path.join(config.data_dir, 'ISIC-2017_Test_v2_Data')
        test_mask_dir = os.path.join(config.data_dir, 'ISIC-2017_Test_v2_Part1_GroundTruth')
        print(f"Expecting ISIC test images in: {test_img_dir}")
        print(f"Expecting ISIC test masks in: {test_mask_dir}")

        if not os.path.exists(test_img_dir):
             raise FileNotFoundError(f"ISIC test image directory not found: {test_img_dir}")
        if not os.path.exists(test_mask_dir):
             # Allow evaluation without masks if directory doesn't exist
             print(f"Warning: ISIC test mask directory not found: {test_mask_dir}. Masks will not be loaded.")
             val_mask_files = []
             val_mask_ids_from_files = []
        else:
             val_mask_files = sorted(glob(os.path.join(test_mask_dir, '*' + mask_ext)))
             val_mask_ids_from_files = [os.path.splitext(os.path.basename(p))[0].replace('_segmentation', '') for p in val_mask_files]

        val_img_files = sorted(glob(os.path.join(test_img_dir, '*' + img_ext)))
        val_img_ids_all = [os.path.splitext(os.path.basename(p))[0] for p in val_img_files]

        # Use all images found, create dummy mask IDs if masks don't exist
        val_img_ids_filtered = val_img_ids_all
        if val_mask_ids_from_files:
            # Filter only if masks exist
            val_img_ids_filtered = [img_id for img_id in val_img_ids_all if img_id in val_mask_ids_from_files]
            val_mask_ids_filtered = val_img_ids_filtered
            print(f"Found {len(val_img_files)} potential test images, {len(val_mask_files)} masks.")
            print(f"Using {len(val_img_ids_filtered)} images with corresponding masks for evaluation.")
        else:
            # Create None mask_ids if no masks are found
            val_mask_ids_filtered = [None] * len(val_img_ids_filtered)
            print(f"Found {len(val_img_files)} test images. Evaluating without masks.")


        if not val_img_ids_filtered:
             raise ValueError(f"No matching image/mask pairs found for ISIC evaluation.")

        val_dataset = Dataset(
            img_ids=val_img_ids_filtered,
            mask_ids=val_mask_ids_filtered, # Pass the potentially None list
            img_dir=test_img_dir,
            mask_dir=test_mask_dir if val_mask_ids_from_files else "", # Pass empty if no masks
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config.num_classes,
            transform=val_transform)

    elif config.dataset == 'Fetus_Head':
        img_ext = '.png'
        mask_ext = '_Annotation.png'
        test_img_dir = os.path.join(config.data_dir, 'validation_set') # Or 'test_set'
        test_mask_dir = os.path.join(config.data_dir, 'validation_set')# Or 'test_set'
        print(f"Expecting Fetus Head test/validation images in: {test_img_dir}")
        print(f"Expecting Fetus Head test/validation masks in: {test_mask_dir}")

        if not os.path.exists(test_img_dir):
             raise FileNotFoundError(f"Fetus Head directory not found: {test_img_dir}")
        if not os.path.exists(test_mask_dir):
            print(f"Warning: Fetus Head mask directory not found: {test_mask_dir}. Cannot load masks.")
            val_mask_files = []
            val_mask_ids_from_files = []
        else:
            val_mask_files = sorted(glob(os.path.join(test_mask_dir, '*' + mask_ext)))
            val_mask_ids_from_files = [os.path.splitext(os.path.basename(p))[0].replace('_Annotation', '') for p in val_mask_files]


        val_img_files = sorted([f for f in glob(os.path.join(test_img_dir, '*' + img_ext))
                               if not f.endswith(mask_ext)])
        val_img_ids_all = [os.path.splitext(os.path.basename(p))[0] for p in val_img_files]

        val_img_ids_filtered = val_img_ids_all
        if val_mask_ids_from_files:
             val_img_ids_filtered = [img_id for img_id in val_img_ids_all if img_id in val_mask_ids_from_files]
             val_mask_ids_filtered = val_img_ids_filtered
             print(f"Found {len(val_img_files)} potential test images, {len(val_mask_files)} masks.")
             print(f"Using {len(val_img_ids_filtered)} images with corresponding masks for evaluation.")
        else:
             val_mask_ids_filtered = [None] * len(val_img_ids_filtered)
             print(f"Found {len(val_img_files)} test images. Evaluating without masks.")


        if not val_img_ids_filtered:
             raise ValueError(f"No matching image/mask pairs found for Fetus Head evaluation.")

        val_dataset = Dataset(
            img_ids=val_img_ids_filtered,
            mask_ids=val_mask_ids_filtered,
            img_dir=test_img_dir,
            mask_dir=test_mask_dir if val_mask_ids_from_files else "",
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config.num_classes,
            transform=val_transform)

    else:
        raise ValueError(f"Unsupported dataset for evaluation: {config.dataset}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False)

    # --- 5. Create Model Instance ---
    print(f"Creating model '{config.arch}' with {config.input_channels} input channels...")
    # Uses the arch_module imported earlier based on config.upgrade
    expected_img_size_for_checkpoint = 256
    model = arch_module.__dict__[config.arch](
                num_classes=config.num_classes,
                input_channels=config.input_channels,
                deep_supervision=config.deep_supervision,
                img_size=expected_img_size_for_checkpoint,
                embed_dims=config.input_list, # Use the parsed list
                no_kan=config.no_kan
                )
    model = model.to(device)

    # --- 6. Load Weights ---
    print(f"Loading model weights from: {config.model_path}")
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    print("Model weights loaded successfully.")

    # --- 7. Set up Criterion (Optional) ---
    if config.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif config.loss:
        try:
            criterion = losses.__dict__[config.loss]().to(device)
        except KeyError:
            print(f"Warning: Loss function '{config.loss}' not found. Proceeding without loss calculation.")
            criterion = None
    else:
        print("INFO: No loss function specified (--loss). Proceeding without loss calculation.")
        criterion = None # Explicitly set to None

    # --- 8. Evaluate ---
    results = evaluate(config, model, val_loader, criterion, device) # Pass config and device

    # --- 9. Print Results ---
    print(f"\nEvaluation Results for {config.dataset} using model {config.model_path}:")
    for metric_name, metric_value in results.items():
        print(f"  Average {metric_name}: {metric_value:.4f}")