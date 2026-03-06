# Crop Delineation from Satellite Imagery

Detect agricultural fields in Sentinel-2 satellite imagery using deep learning — trained on Indian farmland across 6 regions.

## Overview

End-to-end pipeline for binary crop field segmentation:

1. **Data Acquisition** — Sentinel-2 imagery (Planetary Computer)
2. **Mask Generation** — Unsupervised field boundaries via NDVI thresholding and morphological operations
3. **Preprocessing** — Tile imagery and masks into 256×256 patches
4. **Training** — U-Net with EfficientNet-B0 encoder, 4-band input (RGB + NIR)
5. **Prediction** — Field mask generation with sliding window for full satellite images
6. **Evaluation** — IoU, Dice, Precision, Recall + NDVI visualizations

## Architecture

```
Sentinel-2 RGB+NIR (10m) ───► NDVI Thresholding ──► Synthetic Masks ──┐
                                                                    ├──► U-Net ──► Crop Field Mask
Sentinel-2 RGB+NIR (10m) ───────────────────────────────────────────┘
```

| Component   | Choice                   | Why                                          |
| ----------- | ------------------------ | -------------------------------------------- |
| Imagery     | Sentinel-2 L2A (RGB+NIR) | Free, 10m resolution, NIR detects vegetation |
| Labels      | Synthetic (NDVI > 0.3)   | OSM farmland data is too sparse in India     |
| Model       | U-Net                    | Proven for segmentation                      |
| Encoder     | EfficientNet-B0          | Fast, works on CPU                           |
| Loss        | Dice + BCE               | Handles field/non-field class imbalance      |
| Framework   | PyTorch + smp            | Industry standard                            |
| Data Source | Planetary Computer       | Free STAC API, no account needed             |

## Quick Start

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt

# 2. Download imagery for Indian agricultural regions
python download_data.py --regions all --steps download_sentinel

# 3. Generate field masks & format data
python generate_masks.py

# 4. Train (EfficientNet-B0 default, 4-band RGB+NIR)
python train.py --data_dir data/combined --epochs 50

# Train with RGB only (3 bands)
python train.py --data_dir data/combined --in_channels 3 --epochs 50

# 5. Predict field masks
python predict.py --model checkpoints/best_model.pth --input data/punjab/sentinel2_rgbnir.tif

# Quick test (no data needed)
python train.py --dry_run
```

## Dataset Note

Originally, this pipeline was designed to use OpenStreetMap (OSM) `landuse=farmland` polygons as training labels. However, OSM farmland coverage in India is practically zero.
Instead, we now generate unsupervised training masks using NDVI thresholding and morphological operations (`generate_masks.py`).
Alternatively, you can download a pre-labeled European dataset (AI4Boundaries) using `download_ai4b.py`.

## Project Structure

```
crop-delineation/
├── README.md                       # This file
├── HOW_TO_RUN.md                   # Detailed step-by-step guide
├── requirements.txt                # Python dependencies
│
├── download_data.py                # Step 1: Multi-region imagery download
├── generate_masks.py               # Step 2: Unsupervised mask generation
├── train.py                        # Step 3: U-Net training
├── predict.py                      # Step 4: Field mask predictions
├── download_ai4b.py                # Optional: Download AI4Boundaries dataset
│
├── data/                           # Downloaded + processed data
│   ├── punjab/                     # Per-region data
│   ├── haryana/
│   ├── rajasthan/
│   ├── gujarat/
│   ├── madhya_pradesh/
│   ├── up_west/
│   └── combined/                   # Merged train/val split
│       ├── train/images/ & masks/
│       └── val/images/ & masks/
│
├── predictions/                    # Field mask outputs
│
└── checkpoints/                    # Training outputs
    ├── best_model.pth
    ├── training_curves.png
    └── sample_predictions.png
```

## Training Regions

| Region                     | Area          | Agriculture Type               |
| -------------------------- | ------------- | ------------------------------ |
| Punjab (Ludhiana)          | North India   | Wheat, rice, intensive farming |
| Haryana (Karnal)           | North India   | Rice, wheat, flat plains       |
| Rajasthan (Sri Ganganagar) | NW India      | Canal-irrigated, wheat         |
| Gujarat (Anand)            | West India    | Cotton, groundnut, tobacco     |
| MP (Hoshangabad)           | Central India | Soybean, wheat                 |
| UP West (Meerut)           | North India   | Sugarcane, wheat, rice         |

```bash
# Download specific regions:
python download_data.py --regions punjab haryana gujarat

# Custom area (anywhere):
python download_data.py --bbox 72.50 22.10 72.70 22.30 --name anand_farms
```

## Prediction

```bash
# Predict on a satellite image:
python predict.py --model checkpoints/best_model.pth --input data/punjab/sentinel2_rgbnir.tif

# Predict on a folder of tiles:
python predict.py --model checkpoints/best_model.pth --input data/combined/val/images/

# Adjust threshold (lower = more fields detected):
python predict.py --model checkpoints/best_model.pth --input image.tif --threshold 0.4
```

Outputs per image: GeoTIFF mask (`.tif`) + multi-panel visualization (`.png`) with NDVI.

## Key Differences from Road Detection

| Aspect          | Road Detection   | Crop Delineation    |
| --------------- | ---------------- | ------------------- |
| Input bands     | 3 (RGB)          | 4 (RGB + NIR)       |
| Labels source   | OSM highways     | NDVI Thresholding   |
| Label geometry  | Lines → buffered | Synthetic polygons  |
| Loss function   | Dice only        | Dice + BCE combined |
| Encoder default | ResNet-34        | EfficientNet-B0     |
| Visualization   | RGB + mask       | RGB + NDVI + mask   |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) — U-Net implementation
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — free Sentinel-2 access
- [OpenStreetMap](https://www.openstreetmap.org/) — farmland labels
