"""
generate_masks.py — Generate field masks from Sentinel-2 imagery using NDVI
=============================================================================
Since OSM farmland data is too sparse in India, this script creates training
masks by detecting vegetation (high NDVI) and segmenting it into field-like
regions using morphological operations.

Usage:
    python generate_masks.py
    python generate_masks.py --data_dir data --threshold 0.3
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
import glob


def compute_ndvi(image_path):
    """Compute NDVI from a 4-band Sentinel-2 image (R,G,B,NIR)."""
    with rasterio.open(image_path) as src:
        bands = src.read().astype(np.float32)
        profile = src.profile.copy()

    # Band order: R=0, G=1, B=2, NIR=3
    red = bands[0]
    nir = bands[3]

    # Avoid division by zero
    denom = nir + red
    denom[denom == 0] = 1e-10
    ndvi = (nir - red) / denom

    return ndvi, profile


def generate_field_mask(ndvi, ndvi_threshold=0.3, min_field_size=500):
    """
    Generate binary field mask from NDVI.
    
    Steps:
    1. Threshold NDVI to get vegetation mask
    2. Morphological closing to fill small gaps
    3. Remove small objects (noise)
    4. Morphological opening to smooth edges
    """
    # Step 1: NDVI threshold — vegetation pixels
    veg_mask = (ndvi > ndvi_threshold).astype(np.uint8)

    # Step 2: Close small gaps within fields
    struct = ndimage.generate_binary_structure(2, 2)
    veg_mask = ndimage.binary_closing(veg_mask, structure=struct, iterations=3).astype(np.uint8)

    # Step 3: Remove small objects (noise, isolated trees)
    labeled, num_features = ndimage.label(veg_mask)
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_field_size:
            veg_mask[labeled == i] = 0

    # Step 4: Smooth edges
    veg_mask = ndimage.binary_opening(veg_mask, structure=struct, iterations=1).astype(np.uint8)

    return veg_mask


def process_region(region_dir, ndvi_threshold=0.3, min_field_size=500):
    """Generate mask for a single region."""
    image_path = os.path.join(region_dir, "sentinel2_rgbnir.tif")
    mask_path = os.path.join(region_dir, "farmland_mask.tif")

    if not os.path.exists(image_path):
        return False

    print(f"  Computing NDVI...")
    ndvi, profile = compute_ndvi(image_path)

    print(f"  NDVI stats: min={ndvi.min():.3f}, max={ndvi.max():.3f}, mean={ndvi.mean():.3f}")

    print(f"  Generating field mask (threshold={ndvi_threshold})...")
    mask = generate_field_mask(ndvi, ndvi_threshold, min_field_size)

    coverage = mask.sum() / mask.size * 100
    print(f"  Field coverage: {coverage:.2f}%")

    # Save mask
    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8", compress="lzw")
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask, 1)

    print(f"  Saved: {mask_path}")
    return True


def tile_with_masks(data_dir, tile_size=256, stride=128):
    """Re-tile all regions with the new NDVI-based masks."""
    # Import tiling function from download_data
    import sys
    sys.path.insert(0, ".")
    from download_data import tile_region, merge_tiles, REGION_CONFIGS

    regions = []
    total_tiles = 0

    for region_key in REGION_CONFIGS:
        region_dir = os.path.join(data_dir, region_key)
        image_path = os.path.join(region_dir, "sentinel2_rgbnir.tif")
        mask_path = os.path.join(region_dir, "farmland_mask.tif")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        # Check mask has content
        with rasterio.open(mask_path) as src:
            m = src.read(1)
            if m.sum() == 0:
                print(f"  SKIP {region_key}: empty mask")
                continue

        print(f"\n  Tiling {region_key}...")
        tile_dir = os.path.join(region_dir, "tiles")
        n = tile_region(image_path, mask_path, tile_dir, tile_size, stride)
        total_tiles += n
        if n > 0:
            regions.append(region_key)
        print(f"  {region_key}: {n} tiles")

    if total_tiles > 0:
        print(f"\n  Merging {total_tiles} tiles from {len(regions)} regions...")
        out_dir = os.path.join(data_dir, "combined")
        merge_tiles(data_dir, regions, out_dir)

    return total_tiles


def main():
    parser = argparse.ArgumentParser(description="Generate field masks from NDVI")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="NDVI threshold for vegetation detection")
    parser.add_argument("--min_field_size", type=int, default=500,
                        help="Minimum field size in pixels")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--skip_tiling", action="store_true")
    args = parser.parse_args()

    regions_found = 0
    for d in sorted(os.listdir(args.data_dir)):
        region_dir = os.path.join(args.data_dir, d)
        if not os.path.isdir(region_dir) or d == "combined":
            continue
        img_path = os.path.join(region_dir, "sentinel2_rgbnir.tif")
        if not os.path.exists(img_path):
            continue

        print(f"\n{'='*50}")
        print(f"  {d}")
        print(f"{'='*50}")
        if process_region(region_dir, args.threshold, args.min_field_size):
            regions_found += 1

    if regions_found == 0:
        print("\nNo regions with Sentinel-2 imagery found!")
        print("  Run: python download_data.py --regions all --steps download_sentinel")
        return

    print(f"\nGenerated masks for {regions_found} regions")

    if not args.skip_tiling:
        print("\nTiling...")
        n = tile_with_masks(args.data_dir, args.tile_size, args.stride)
        if n > 0:
            print(f"\nDONE! {n} tiles ready.")
            print(f"  python train.py --data_dir {args.data_dir}/combined --epochs 50")
        else:
            print("\nNo tiles generated. Try lowering --threshold.")
    else:
        print("\nSkipped tiling. Run with --skip_tiling=false to tile.")


if __name__ == "__main__":
    main()
