# How to Run — Crop Delineation Pipeline

Step-by-step instructions to set up and run the crop field detection pipeline.

## Prerequisites

- **Python 3.9+** installed
- **Git** (optional, for version control)
- ~2GB disk space for data + model
- Internet connection (for downloading satellite imagery)

---

## Step 1: Setup Environment

```bash
# Navigate to project
cd "d:\Internship related\GJ-Map Solutions\crop-delineation"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Quick Test (No Data Needed)

Before downloading any real data, verify everything works:

```bash
python train.py --dry_run
```

This trains for 5 epochs on synthetic rectangular fields. You should see:

- Loss decreasing
- IoU/Dice improving
- `checkpoints/training_curves.png` and `checkpoints/sample_predictions.png` generated

---

## Step 3: Download Training Data

```bash
# Download ALL 6 regions (takes 10-30 mins depending on internet):
python download_data.py --regions all

# Or start with just one region to test:
python download_data.py --regions punjab
```

This downloads:

1. Sentinel-2 imagery (4-band: RGB + NIR) from Microsoft Planetary Computer
2. Saves them locally by region

**Note:** Previously this also downloaded OpenStreetMap farmland data, but OSM coverage for agriculture in India is practically zero. You MUST run Step 4 to generate training masks if using this method.

**Custom area:**

```bash
python download_data.py --bbox 72.50 22.10 72.70 22.30 --name my_area
```

Use [bboxfinder.com](http://bboxfinder.com) to find coordinates.

---

## Step 4: Generate Masks & Tile Data

Since OSM data is missing, we generate training masks directly from the Sentinel-2 imagery using NDVI (Normalized Difference Vegetation Index) thresholding:

```bash
python generate_masks.py --threshold 0.3 --min_field_size 500
```

This will:

1. Detect vegetation (NDVI > 0.3)
2. Use morphological operations to clean noise and create field boundary shapes
3. Cut everything into 256×256 patches
4. Merge all regions into `data/combined/train` and `val`

> **Note:** If you are using `download_ai4b.py` (Step 3.5), you DO NOT need to run this step.

---

## Step 4.5: AI4Boundaries Dataset (Recommended)

Instead of relying on synthetic NDVI masks (Steps 3 & 4), it is highly recommended to use the massive pre-labeled AI4Boundaries European dataset for robust training.

```bash
# Requires netCDF4
pip install netCDF4

# Download full training set (22GB) and convert to our format
python download_ai4b.py --split train
```

This will automatically format images and binary extent masks into `data/combined/train` and `data/combined/val`, making it ready for `train.py`.

---

## Step 5: Train the Model

```bash
# Full training (default: EfficientNet-B0, 4-band RGB+NIR, 50 epochs)
python train.py --data_dir data/combined --epochs 50

# RGB only (3 bands, if NIR not wanted)
python train.py --data_dir data/combined --in_channels 3 --epochs 50

# Faster training (fewer epochs to verify):
python train.py --data_dir data/combined --epochs 10
```

### Training outputs:

- `checkpoints/best_model.pth` — best model weights
- `checkpoints/training_curves.png` — loss/IoU/Dice plots
- `checkpoints/sample_predictions.png` — visual predictions

### Fine-tuning:

```bash
# Decoder only (faster)
python train.py --data_dir data/combined --finetune checkpoints/best_model.pth --freeze_encoder

# Full fine-tune
python train.py --data_dir data/combined --finetune checkpoints/best_model.pth
```

---

## Step 6: Predict on New Images

```bash
# Single image:
python predict.py --model checkpoints/best_model.pth --input data/punjab/sentinel2_rgbnir.tif

# Folder of tiles:
python predict.py --model checkpoints/best_model.pth --input data/combined/val/images/

# Lower threshold (detect more fields):
python predict.py --model checkpoints/best_model.pth --input image.tif --threshold 0.3
```

### Prediction outputs (per image):

- `predictions/<name>_field_mask.tif` — binary GeoTIFF mask
- `predictions/<name>_visualization.png` — RGB + NDVI + probability + overlay

---

## Step 7: End-to-End Visual Demo

Once the model is trained, use `small_area_demo.py` to test it on a fresh 5x5 km area. This script downloads fresh Sentinel-2 imagery on the fly, passes it through the model, and builds a comprehensive publication-ready report evaluating its performance.

```bash
python small_area_demo.py
```

Outputs in the `small_area_results/` directory:

- `report_results.png` - 6-panel technical breakdown including NDVI and Probability mapping.
- `field_overlay.png` - Clean RGB + mask overlay for easy sharing.

---

## Troubleshooting

| Issue                 | Solution                                                                        |
| --------------------- | ------------------------------------------------------------------------------- |
| `No scenes found`     | Increase `--max_cloud` (default 15%)                                            |
| `0 farmland polygons` | Area may lack OSM data; try a different region                                  |
| Out of memory         | Reduce `--batch_size` (default 8)                                               |
| Slow training         | Use `--encoder efficientnet-b0` (default), reduce `--epochs`                    |
| Import errors         | Ensure venv is activated and `pip install -r requirements.txt` ran successfully |

---

## Project Commands Cheat Sheet

```bash
# Test everything works
python train.py --dry_run

# Recommended Train Flow: Download AI4Boundaries & Train
python download_ai4b.py --split train
python train.py --data_dir data/combined --epochs 50

# Demo the Evaluation Results
python small_area_demo.py

# Predict Custom Images
python predict.py --model checkpoints/best_model.pth --input <image_or_folder>
```
