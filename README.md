# Crop Delineation from Satellite Imagery

Detect agricultural fields in Sentinel-2 satellite imagery using deep learning — trained on Indian farmland across 6 regions.

## Overview

End-to-end pipeline for binary crop field segmentation:

1. **Data Acquisition** — Sentinel-2 imagery (Planetary Computer) or AI4Boundaries dataset
2. **Mask Generation** — Pre-labeled data via `download_ai4b.py` or unsupervised boundaries via NDVI thresholding
3. **Preprocessing** — Tile imagery and masks into 256×256 patches
4. **Training** — U-Net with EfficientNet-B0 encoder, 4-band input (RGB + NIR)
5. **Prediction** — Field mask generation with sliding window for full satellite images
6. **Demo** — End-to-end small area evaluation with visualization (`small_area_demo.py`)
7. **Evaluation** — IoU, Dice, Precision, Recall + NDVI visualizations

## Architecture

```
Sentinel-2 RGB+NIR (10m) ───► NDVI Thresholding ──► Synthetic Masks ──┐
                                                                    ├──► U-Net ──► Crop Field Mask
Sentinel-2 RGB+NIR (10m) ───────────────────────────────────────────┘
```

| Component   | Choice                    | Why                                          |
| ----------- | ------------------------- | -------------------------------------------- |
| Imagery     | Sentinel-2 L2A (RGB+NIR)  | Free, 10m resolution, NIR detects vegetation |
| Labels      | AI4Boundaries / Synthetic | High quality EU dataset or NDVI Thresholding |
| Model       | U-Net                     | Proven for segmentation                      |
| Encoder     | EfficientNet-B0           | Fast, works on CPU                           |
| Loss        | Dice + BCE                | Handles field/non-field class imbalance      |
| Framework   | PyTorch + smp             | Industry standard                            |
| Data Source | JRC FTP / Planetary Comp  | Public datasets with API access              |

## Quick Start

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt

# 2. Download training data
# Option A: AI4Boundaries (Recommended - high quality European dataset)
python download_ai4b.py --split train

# Option B: Synthetic Indian Regions (Requires generate_masks.py afterwards)
python download_data.py --regions all --steps download_sentinel
python generate_masks.py

# 3. Train (EfficientNet-B0 default, 4-band RGB+NIR)
python train.py --data_dir data/combined --epochs 50

# 4. End-to-End Evaluation (Visual Demo)
python small_area_demo.py

# 5. Predict field masks on custom imagery
python predict.py --model checkpoints/best_model.pth --input data/punjab/sentinel2_rgbnir.tif
```

## Dataset Note

The recommended training path is to use the massive pre-labeled European **AI4Boundaries dataset** through `download_ai4b.py`. This provides high-quality labels for model training and automatically converts them to our format.

Alternatively, if studying Indian agriculture specifically, OSM farmland coverage is practically zero. We instead generate unsupervised training masks using NDVI thresholding and morphological operations (`generate_masks.py`) run over imagery collected via `download_data.py`.

## Project Structure

```
crop-delineation/
├── README.md                       # This file
├── HOW_TO_RUN.md                   # Detailed step-by-step guide
├── requirements.txt                # Python dependencies
│
├── download_data.py                # Step 1B: Multi-region imagery download
├── generate_masks.py               # Step 1B: Unsupervised mask generation
├── download_ai4b.py                # Step 1A: Download AI4Boundaries dataset
├── train.py                        # Step 2: U-Net training
├── small_area_demo.py              # Step 3: End-to-end evaluation & visual report
├── predict.py                      # Step 4: Field mask predictions
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
