# UKAN Segmentation Project

This project implements and trains a UKAN (U-Net with Kolmogorov-Arnold Networks) model for semantic segmentation tasks on various medical imaging datasets.

## Features

*   Supports training on ISIC, Lumbar Spine, and Fetus Head datasets.
*   Implements standard UKAN architecture (`archs.py`).
*   Includes an optional **upgraded** UKAN architecture (`archs_upgraded.py`).
*   Saves training logs (`log.csv`), configuration (`config.yml`), and model checkpoints (`model.pth`, `model_epoch_X.pth`).
*   Visualizes segmentation progress during training, saving comparison images (`progress_images/segOutputEpoch_X.png`).
*   Includes an evaluation script (`evaluate.py`) for assessing trained models.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sath-vik/Improving-Medical-Image-Segmentation-Models.git
    cd Improving-Medical-Image-Segmentation-Models
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    *   **On Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
4.  **Install dependencies:**
    ```bash
    # Install PyTorch first (adjust CUDA version if needed)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or your specific CUDA version
    # Install other requirements
    pip install medpy tensorboardX timm pandas numpy PyYAML scikit-learn tqdm albumentations matplotlib scikit-image
    ```
    *(Note: Check the official PyTorch website for the correct installation command based on your OS, package manager, and CUDA version: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))*

## Dataset Preparation

1.  Download the required datasets (ISIC 2017, Lumbar Spine, Fetus Head). Ensure you agree to the terms and conditions of each dataset.

2.  Organize the datasets according to the structure expected by the `train.py` script. The script uses the `--data_dir` argument to locate the base directory for each dataset. The default paths used in the examples are:
    *   `/content/drive/MyDrive/Datasets/ISIC`
    *   `/content/drive/MyDrive/Datasets/Lumbar_Spine/`
    *   `/content/drive/MyDrive/Datasets/Fetus_Head`

3.  **Expected Directory Structure within `--data_dir`:**

    *   **For `--dataset ISIC`:**
        The script expects separate folders for training and validation images/masks within the `data_dir`.
        ```
        <data_dir>/
        ├── ISIC-2017_Training_Data/
        │   ├── ISIC_0000000.jpg
        │   └── ... (training images)
        ├── ISIC-2017_Training_Part1_GroundTruth/
        │   ├── ISIC_0000000_segmentation.png
        │   └── ... (training masks)
        ├── ISIC-2017_Validation_Data/
        │   ├── ISIC_0000002.jpg
        │   └── ... (validation images)
        └── ISIC-2017_Validation_Part1_GroundTruth/
            ├── ISIC_0000002_segmentation.png
            └── ... (validation masks)
        ```
        *(Image files should be `.jpg`, mask files should be `.png` and end with `_segmentation.png`)*

    *   **For `--dataset Lumbar_Spine`:**
        The script expects a specific structure within a `png` subdirectory inside `data_dir`. Images and masks are organized by 'Case'. The script performs an 80/20 train/validation split internally based on the files found.
        ```
        <data_dir>/
        └── png/
            ├── images/
            │   ├── Case_01/
            │   │   ├── Slice_01.png
            │   │   └── ...
            │   └── Case_XX/
            │       └── ... (image slices)
            └── masks/
                ├── Case_01/
                │   ├── Labels_Slice_01.png
                │   └── ...
                └── Case_XX/
                    └── ... (corresponding mask slices)
        ```
        *(Image filenames contain `Slice_YY.png`, corresponding mask filenames must be `Labels_Slice_YY.png`)*

    *   **For `--dataset Fetus_Head`:**
        The script expects separate `training_set` and `validation_set` folders. Within each folder, both images and their corresponding annotation masks reside.
        ```
        <data_dir>/
        ├── training_set/
        │   ├── image_001.png
        │   ├── image_001_Annotation.png
        │   └── ... (training images and masks together)
        └── validation_set/
            ├── image_101.png
            ├── image_101_Annotation.png
            └── ... (validation images and masks together)
        ```
        *(Image files are `.png`, mask files are `.png` and end with `_Annotation.png`)*

4.  *(Optional)* If your datasets are located elsewhere, you will need to adjust the `--data_dir` argument in the training and evaluation commands accordingly.

## Training

Use the `train.py` script to train the model. Select the dataset using the `--dataset` argument and provide the correct path using `--data_dir`.

**Using Standard UKAN Architecture (`archs.py`):**

*   **ISIC Dataset:**
    ```bash
    python train.py --dataset ISIC --data_dir '/content/drive/MyDrive/Datasets/ISIC' --name isic_ukan_standard --epochs 30 -b 8
    ```
*   **Lumbar Spine Dataset:**
    ```bash
    # Note: Lumbar spine uses 1 input channel
    python train.py --dataset Lumbar_Spine --data_dir '/content/drive/MyDrive/Datasets/Lumbar_Spine/' --name lumbar_ukan_standard --epochs 30 -b 8 --input_channels 1
    ```
*   **Fetus Head Dataset:**
    ```bash
    # Note: Fetus Head uses 1 input channel
    python train.py --dataset Fetus_Head --data_dir '/content/drive/MyDrive/Datasets/Fetus_Head' --name fetus_ukan_standard --epochs 30 -b 8 --input_channels 1
    ```

**Using Upgraded UKAN Architecture (`archs_upgraded.py`):**

To use the upgraded architecture defined in `archs_upgraded.py`, simply add the `--upgrade` flag to any of the training commands above. Make sure the `archs_upgraded.py` file exists in the same directory.

*   **Example (ISIC Dataset with Upgraded Arch):**
    ```bash
    python train.py --dataset ISIC --data_dir '/content/drive/MyDrive/Datasets/ISIC' --name isic_ukan_upgraded --epochs 30 -b 8 --upgrade
    ```

*You can customize other arguments like `--epochs`, `--batch_size` (`-b`), `--lr`, `--arch` (if you have multiple models within the files), `--name` (for experiment tracking), etc. Run `python train.py --help` to see all available options.*

## Outputs

During and after training, the following outputs will be generated in the directory specified by `--output_dir` (default: `/content/drive/MyDrive/Codes_ADL_Assignment/Seg_UKAN/outputs`), under a subfolder named using the `--name` argument (e.g., `isic_ukan_standard`):

- **`config.yml`**: The configuration file used for the training run.
- **`log.csv`**: A CSV file containing training and validation metrics per epoch (e.g., loss, IoU, Dice score).
- **`model.pth`**: The model checkpoint corresponding to the best validation IoU achieved during training.
- **`archs_used.py`**: A copy of the architecture file used for the run (`archs.py` or `archs_upgraded.py`).
- **`progress_images/`**: A folder containing sample segmentation visualizations saved during training (e.g., `segOutputEpoch_X.png`).


## Evaluation

Use the `evaluate.py` script to evaluate a trained model checkpoint. You need to specify the path to the saved model (`--model_path`), the dataset (`--dataset`), the data directory (`--data_dir`), and the architecture (`--arch`) used for training. Ensure that the input channels (`--input_channels`), input height (`--input_h`), and input width (`--input_w`) match the training configuration.

**Example:**

For an ISIC model trained with the standard UKAN architecture:

```bash
python evaluate.py \
    --model_path '/content/drive/MyDrive/Codes_ADL_Assignment/Seg_UKAN/outputs/isic_ukan_standard/model.pth' \
    --data_dir '/content/drive/MyDrive/Datasets/ISIC/' \
    --dataset ISIC \
    --arch UKAN \
    --input_channels 3 \
    --input_h 256 \
    --input_w 256
```
Run python evaluate.py --help to see all available evaluation options.