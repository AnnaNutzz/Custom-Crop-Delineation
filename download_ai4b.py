"""
download_ai4b.py — Download AI4Boundaries Sentinel-2 Dataset
==============================================================
Downloads the AI4Boundaries Sentinel-2 test set from the JRC FTP server
and converts it to our training format (images + binary masks).

Usage:
    python download_ai4b.py                    # Download test set (4.7 GB)
    python download_ai4b.py --split train      # Download train set (22 GB)
    python download_ai4b.py --skip_download    # Just convert already downloaded files
"""

import argparse
import os
import sys
import zipfile
import glob
import random
import numpy as np

BASE_URL = "http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/sentinel2"


def download_file(url, dst_path):
    """Download a file with progress bar, using browser-like headers."""
    import urllib.request

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    req = urllib.request.Request(url, headers=headers)

    print(f"  Downloading: {os.path.basename(dst_path)}")
    print(f"  From: {url}")

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            total = int(response.headers.get("Content-Length", 0))
            total_mb = total / (1024 * 1024) if total else 0

            with open(dst_path, "wb") as f:
                downloaded = 0
                block_size = 1024 * 1024  # 1 MB chunks
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {downloaded/(1024*1024):.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)
                print()
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(f"\n  If automated download fails, manually download from:")
        print(f"    {url}")
        print(f"  And save to: {dst_path}")
        return False


def download_dataset(data_dir, split="test"):
    """Download AI4Boundaries zip file for the given split."""
    raw_dir = os.path.join(data_dir, "ai4b_raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_name = f"{split}.zip"
    zip_path = os.path.join(raw_dir, zip_name)
    zip_url = f"{BASE_URL}/{zip_name}"

    masks_path = os.path.join(raw_dir, "masks.zip")
    masks_url = f"{BASE_URL}/masks.zip"

    # Download the split zip
    if not os.path.exists(zip_path):
        print(f"\n{'='*60}")
        print(f"  Downloading AI4Boundaries {split} set...")
        print(f"{'='*60}")
        success = download_file(zip_url, zip_path)
        if not success:
            return None
    else:
        print(f"  {zip_name} already exists, skipping download")

    # Download masks
    if not os.path.exists(masks_path):
        print(f"\n  Downloading masks...")
        success = download_file(masks_url, masks_path)
        if not success:
            return None
    else:
        print(f"  masks.zip already exists, skipping download")

    # Extract
    extract_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"\n  Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print(f"  Extracting masks.zip...")
        with zipfile.ZipFile(masks_path, "r") as z:
            z.extractall(extract_dir)
        print("  Extraction complete!")

    return extract_dir


def convert_to_training_format(extract_dir, out_dir, val_split=0.15, max_samples=None):
    """
    Convert AI4Boundaries to our training format.

    AI4Boundaries Sentinel-2 format:
      - Images: .nc (NetCDF) files with bands R, G, B, NIR, NDVI x 6 months
      - Masks: .tif files with 4 bands (extent, boundary, distance, field_id)
    
    Our format:
      - images/tile_XXXXX.tif  (4 bands: R, G, B, NIR)
      - masks/tile_XXXXX.tif   (1 band: binary extent)
    """
    import rasterio
    try:
        import netCDF4 as nc
    except ImportError:
        print("ERROR: netCDF4 not installed. Run: pip install netCDF4")
        return 0

    # Find all files
    nc_files = sorted(glob.glob(os.path.join(extract_dir, "**", "*.nc"), recursive=True))
    tif_files = sorted(glob.glob(os.path.join(extract_dir, "**", "*label*.tif"), recursive=True))
    
    if not tif_files:
        tif_files = sorted(glob.glob(os.path.join(extract_dir, "**", "*.tif"), recursive=True))

    print(f"\n  Found {len(nc_files)} image files (.nc)")
    print(f"  Found {len(tif_files)} mask files (.tif)")

    if not nc_files or not tif_files:
        print("\n  Listing files found:")
        for root, dirs, files in os.walk(extract_dir):
            for f in files[:20]:
                print(f"    {os.path.relpath(os.path.join(root, f), extract_dir)}")
        return 0

    # Match by sample ID (e.g., "AT_12345")
    def get_id(path):
        base = os.path.basename(path)
        parts = base.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return base.split(".")[0]

    mask_by_id = {get_id(f): f for f in tif_files}

    pairs = []
    for nc_file in nc_files:
        sid = get_id(nc_file)
        if sid in mask_by_id:
            pairs.append((nc_file, mask_by_id[sid]))

    print(f"  Matched pairs: {len(pairs)}")

    if max_samples and len(pairs) > max_samples:
        random.shuffle(pairs)
        pairs = pairs[:max_samples]
        print(f"  Using subset: {max_samples} samples")

    # Split
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    converted = 0
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(out_dir, split_name, "images")
        msk_dir = os.path.join(out_dir, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        for i, (nc_path, tif_path) in enumerate(split_pairs):
            tile_name = f"tile_{converted:05d}.tif"
            try:
                # Read NetCDF — get R, G, B, NIR for a single month
                ds = nc.Dataset(nc_path)
                var_names = list(ds.variables.keys())

                # Try common band naming conventions
                img = None
                # AI4Boundaries uses B2,B3,B4,B8; others may use B02,B03,B04,B08
                r_name = "B4" if "B4" in var_names else ("B04" if "B04" in var_names else None)
                g_name = "B3" if "B3" in var_names else ("B03" if "B03" in var_names else None)
                b_name = "B2" if "B2" in var_names else ("B02" if "B02" in var_names else None)
                nir_name = "B8" if "B8" in var_names else ("B08" if "B08" in var_names else None)
                
                if r_name and g_name and b_name and nir_name:
                    r = np.array(ds[r_name][:], dtype=np.float32)
                    g = np.array(ds[g_name][:], dtype=np.float32)
                    b = np.array(ds[b_name][:], dtype=np.float32)
                    nir = np.array(ds[nir_name][:], dtype=np.float32)
                    if r.ndim == 3:  # (time, H, W) — pick middle time step
                        mid = r.shape[0] // 2
                        r, g, b, nir = r[mid], g[mid], b[mid], nir[mid]
                    img = np.stack([r, g, b, nir], axis=0)
                else:
                    # Generic: pick first data variable
                    data_vars = [v for v in var_names if v not in ("lat", "lon", "x", "y", "time", "crs", "spatial_ref")]
                    if data_vars:
                        data = np.array(ds[data_vars[0]][:], dtype=np.float32)
                        if data.ndim == 4:  # (time, bands, H, W)
                            data = data[data.shape[0] // 2]
                        if data.ndim == 3 and data.shape[0] >= 4:
                            img = data[:4]
                        elif data.ndim == 3:
                            img = data[:3]
                            img = np.concatenate([img, np.zeros_like(img[:1])], axis=0)
                ds.close()

                if img is None:
                    continue

                # Read mask — extent minus boundary = field interiors with edges removed
                # Band 1 = extent (field/non-field), Band 2 = boundary (field edges)
                with rasterio.open(tif_path) as src:
                    extent = (src.read(1) > 0).astype(np.uint8)
                    boundary = (src.read(2) > 0).astype(np.uint8)
                    mask = extent - (extent * boundary)  # Interior only, edges = 0

                # Ensure matching dimensions
                h, w = min(img.shape[1], mask.shape[0]), min(img.shape[2], mask.shape[1])
                img = img[:, :h, :w]
                mask = mask[:h, :w]

                # Normalize image to [0, 1] range
                for b_idx in range(img.shape[0]):
                    band = img[b_idx]
                    p98 = np.percentile(band, 98)
                    if p98 > 0:
                        img[b_idx] = band / p98
                img = np.clip(img, 0, 1)

                # Save image
                profile = {"driver": "GTiff", "dtype": "float32",
                           "width": w, "height": h, "count": 4, "compress": "lzw"}
                with rasterio.open(os.path.join(img_dir, tile_name), "w", **profile) as dst:
                    dst.write(img)

                # Save mask
                msk_profile = {"driver": "GTiff", "dtype": "uint8",
                               "width": w, "height": h, "count": 1, "compress": "lzw"}
                with rasterio.open(os.path.join(msk_dir, tile_name), "w", **msk_profile) as dst:
                    dst.write(mask, 1)

                converted += 1
                if converted % 100 == 0:
                    print(f"    Converted {converted} samples...")

            except Exception as e:
                print(f"    WARN: Skipped {os.path.basename(nc_path)}: {e}")
                continue

    print(f"\n  Total converted: {converted}")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Download AI4Boundaries dataset")
    parser.add_argument("--data_dir", default="data", help="Base data directory")
    parser.add_argument("--split", default="test", choices=["train", "test", "val"],
                        help="Which split to download (test=4.7GB, train=22GB)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to convert")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, just convert existing files")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "ai4b_raw")
    out_dir = os.path.join(args.data_dir, "combined")

    if not args.skip_download:
        extract_dir = download_dataset(args.data_dir, args.split)
        if extract_dir is None:
            print("\nDownload failed. Try downloading manually from browser:")
            print(" main url: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/")
            print(" test url: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/sentinel2/test.zip")
            print(" mask url: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/masks.zip")
            print(f"Save both to: {raw_dir}")
            print(f"Then run: python download_ai4b.py --skip_download")
            return
    else:
        extract_dir = raw_dir  # Files extracted directly here (test/, masks/)
        if not os.path.exists(extract_dir):
            print(f"ERROR: {extract_dir} not found. Download first.")
            return

    print("\nConverting to training format...")
    n = convert_to_training_format(extract_dir, out_dir, args.val_split, args.max_samples)

    if n > 0:
        print(f"\nDONE! {n} samples ready for training.")
        print(f"  python train.py --data_dir {out_dir} --epochs 50")
    else:
        print("\nERROR: No samples converted. Check the extracted files.")


if __name__ == "__main__":
    main()
