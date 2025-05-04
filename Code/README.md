# UKAN Segmentation Project

This project implements and trains a UKAN (U-Net with Kolmogorov-Arnold Networks) model for semantic segmentation tasks on various medical imaging datasets.

## Features

*   Supports training on ISIC, Lumbar Spine, and Fetus Head datasets.
*   Implements standard UKAN architecture.
*   Includes an optional **upgraded** UKAN architecture (in `archs_upgraded.py`).
*   Saves training logs and model checkpoints.
*   Visualizes segmentation progress during training.

## Setup
1.  **Install dependencies:**
    ```bash
    pip install medpy tensorboardX timm pandas numpy PyYAML scikit-learn tqdm albumentations matplotlib scikit-image
    ```

## Dataset Preparation

1.  Download the required datasets (ISIC, Lumbar Spine, Fetus Head).
2.  Organize the datasets. The training script expects the data to be located in paths like:
    *   `/content/drive/MyDrive/Datasets/ISIC`
    *   `/content/drive/MyDrive/Datasets/Lumbar_Spine/`
    *   `/content/drive/MyDrive/Datasets/Fetus_Head`
    *(Adjust the `--data_dir` argument in the training commands if your datasets are located elsewhere).*
3.  Ensure the directory structure within each dataset folder matches what the `Dataset` or `LumbarSpineDataset` classes expect (e.g., separate image and mask folders, specific naming conventions). Refer to `dataset.py` and `lumbarSpineDataset.py` for details if needed.

## Training

Use the `train.py` script to train the model. Select the dataset using the `--dataset` argument and provide the correct path using `--data_dir`.

**Using Standard UKAN Architecture (`archs.py`):**

*   **ISIC Dataset:**
    ```bash
    python train.py --dataset ISIC --data_dir '/content/drive/MyDrive/Datasets/ISIC' --name isic_ukan_standard --epochs 30 -b 8
    ```
*   **Lumbar Spine Dataset:**
    ```bash
    python train.py --dataset Lumbar_Spine --data_dir '/content/drive/MyDrive/Datasets/Lumbar_Spine/' --name lumbar_ukan_standard --epochs 30 -b 8 --input_channels 1
    ```
*   **Fetus Head Dataset:**
    ```bash
    python train.py --dataset Fetus_Head --data_dir '/content/drive/MyDrive/Datasets/Fetus_Head' --name fetus_ukan_standard --epochs 30 -b 8 --input_channels 1
    ```

**Using Upgraded UKAN Architecture (`archs_upgraded.py`):**

To use the upgraded architecture defined in `archs_upgraded.py`, simply add the `--upgrade` flag to any of the training commands above. Make sure the `archs_upgraded.py` file exists in the same directory.

*   **Example (ISIC Dataset with Upgraded Arch):**
    ```bash
    python train.py --dataset ISIC --data_dir '/content/drive/MyDrive/Datasets/ISIC' --name isic_ukan_upgraded --epochs 30 -b 8 --upgrade
    ```

*You can customize other arguments like `--epochs`, `--batch_size` (`-b`), `--lr`, `--arch` (if you have multiple models within the files), `--name` (for experiment tracking), etc.*

## Evaluation

Use the `evaluate.py` script to evaluate a trained model checkpoint.
Eg :

```bash
python evaluate.py \
    --model_path '/content/drive/MyDrive/Codes_ADL_Assignment/Seg_UKAN/outputs/fin_is/model.pth' \
    --data_dir '/content/drive/MyDrive/Datasets/ISIC/' \
    --dataset ISIC \
    --arch UKAN \
    --input_channels 3 \
    --input_h 256 \
    --input_w 256