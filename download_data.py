"""
download_data.py — Download Sentinel-2 + OSM Farmland for Indian Regions
=========================================================================
Downloads training data for crop field delineation.

Uses:
  - Microsoft Planetary Computer (free) for Sentinel-2 L2A (RGB + NIR)
  - Overpass API (free) for OSM farmland polygons

Supported regions: punjab, haryana, rajasthan, gujarat, madhya_pradesh, up_west
                   (agricultural areas in India)

Usage:
    # Download all regions:
    python download_data.py --regions all

    # Download specific regions:
    python download_data.py --regions punjab haryana

    # Download one region:
    python download_data.py --regions gujarat

    # Download ANY custom area:
    python download_data.py --bbox 72.50 22.10 72.70 22.30 --name anand_farms

    # Specific processing steps only:
    python download_data.py --regions punjab --steps download_sentinel download_osm
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box, mapping
import requests


# ─── Region Bounding Boxes (west, south, east, north) — WGS84 ───
# These target *agricultural* areas, not city centres.

REGION_CONFIGS = {
    "punjab": {
        "name": "Punjab (Ludhiana)",
        "bbox": (75.50, 30.60, 76.10, 31.15),
        "utm_crs": "EPSG:32643",
    },
    "haryana": {
        "name": "Haryana (Karnal)",
        "bbox": (76.70, 29.45, 77.30, 30.00),
        "utm_crs": "EPSG:32643",
    },
    "rajasthan": {
        "name": "Rajasthan (Sri Ganganagar)",
        "bbox": (73.50, 29.65, 74.15, 30.20),
        "utm_crs": "EPSG:32643",
    },
    "gujarat": {
        "name": "Gujarat (Anand)",
        "bbox": (72.70, 22.30, 73.30, 22.85),
        "utm_crs": "EPSG:32643",
    },
    "madhya_pradesh": {
        "name": "MP (Hoshangabad)",
        "bbox": (77.45, 22.45, 78.05, 23.00),
        "utm_crs": "EPSG:32644",
    },
    "up_west": {
        "name": "UP West (Meerut)",
        "bbox": (77.40, 28.75, 78.00, 29.30),
        "utm_crs": "EPSG:32644",
    },
}


def get_utm_crs(lon):
    """Auto-detect UTM zone EPSG code from longitude."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}"


# ═══════════════════════════════════════════════════════════════
# 1. Download Sentinel-2 (RGB + NIR)
# ═══════════════════════════════════════════════════════════════

def download_sentinel2(region_key, bbox, output_dir, max_cloud=15):
    """Download Sentinel-2 RGB + NIR bands from Planetary Computer."""
    from pystac_client import Client
    import planetary_computer
    from pyproj import Transformer
    from rasterio.mask import mask as rio_mask

    config = REGION_CONFIGS[region_key]
    print(f"\n  [Sentinel-2] Querying for {config['name']}...")

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for recent, low-cloud images
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=["-properties.datetime"],
        max_items=5,
    )

    items = list(search.items())
    if not items:
        print(f"  ERROR: No scenes found for {region_key}. Try --max_cloud higher.")
        return None

    item = items[0]
    print(f"  Scene: {item.id}")
    print(f"  Date: {item.properties['datetime'][:10]}")
    print(f"  Cloud: {item.properties['eo:cloud_cover']:.1f}%")

    # Download 4 bands: Red, Green, Blue, NIR
    band_names = ["B04", "B03", "B02", "B08"]  # R, G, B, NIR
    bands = []
    out_transform = None
    out_crs = None

    for band_name in band_names:
        asset = item.assets[band_name]
        print(f"  Downloading {band_name}...", end=" ", flush=True)

        with rasterio.open(asset.href) as src:
            # Reproject bbox to raster's CRS
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            west, south = transformer.transform(bbox[0], bbox[1])
            east, north = transformer.transform(bbox[2], bbox[3])
            aoi_utm = box(west, south, east, north)

            data, transform = rio_mask(src, [mapping(aoi_utm)], crop=True)
            bands.append(data[0])
            out_transform = transform
            out_crs = src.crs
            profile = src.profile.copy()
            print(f"OK ({data.shape[1]}x{data.shape[2]})")

    stacked = np.stack(bands, axis=0)  # (4, H, W)
    h, w = stacked.shape[1], stacked.shape[2]

    out_path = os.path.join(output_dir, "sentinel2_rgbnir.tif")
    profile.update(
        driver="GTiff", count=4, height=h, width=w,
        transform=out_transform, crs=out_crs, compress="lzw",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    print(f"  Saved: {out_path} ({w}x{h} px, 4 bands: RGB+NIR)")
    return out_path


# ═══════════════════════════════════════════════════════════════
# 2. Download OSM Farmland Polygons
# ═══════════════════════════════════════════════════════════════

def download_osm_farmland(region_key, bbox, output_dir):
    """Download farmland polygons from OSM Overpass API."""
    config = REGION_CONFIGS[region_key]
    print(f"\n  [OSM] Downloading farmland for {config['name']}...")

    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]

    # Query for agricultural landuse polygons
    query = f"""[out:json][timeout:300];
(
  way["landuse"="farmland"]({south},{west},{north},{east});
  way["landuse"="farm"]({south},{west},{north},{east});
  way["landuse"="orchard"]({south},{west},{north},{east});
  way["landuse"="vineyard"]({south},{west},{north},{east});
  relation["landuse"="farmland"]({south},{west},{north},{east});
  relation["landuse"="farm"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=300,
    )

    if response.status_code != 200:
        print(f"  ERROR: Overpass returned {response.status_code}")
        return None

    data = response.json()

    # Two-pass parsing: collect nodes first, then build polygons
    nodes = {}
    raw_ways = []
    raw_relations = []

    for element in data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element["type"] == "way":
            raw_ways.append(element)
        elif element["type"] == "relation":
            raw_relations.append(element)

    # Build way geometries (as lookup for relations)
    way_coords = {}
    for element in raw_ways:
        coords = [nodes[nid] for nid in element.get("nodes", []) if nid in nodes]
        if len(coords) >= 3:
            way_coords[element["id"]] = coords

    features = []

    # Simple ways → polygons
    for element in raw_ways:
        coords = [nodes[nid] for nid in element.get("nodes", []) if nid in nodes]
        if len(coords) >= 4:  # Need 4+ points for a polygon (first == last)
            # Close the ring if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": element.get("tags", {}),
            })

    # Relations → multipolygons
    for rel in raw_relations:
        outer_rings = []
        for member in rel.get("members", []):
            if member.get("role") == "outer" and member.get("ref") in way_coords:
                ring = way_coords[member["ref"]]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                outer_rings.append(ring)

        for ring in outer_rings:
            if len(ring) >= 4:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": rel.get("tags", {}),
                })

    geojson = {"type": "FeatureCollection", "features": features}

    out_path = os.path.join(output_dir, "farmland.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"  Found {len(features)} farmland polygons")
    return out_path


# ═══════════════════════════════════════════════════════════════
# 3. Rasterize Farmland Polygons
# ═══════════════════════════════════════════════════════════════

def rasterize_farmland(region_key, image_path, farmland_path, output_dir):
    """Create binary farmland mask aligned to Sentinel-2 image."""
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    print(f"\n  [Rasterize] Creating farmland mask...")

    with open(farmland_path) as f:
        farmland = json.load(f)

    with rasterio.open(image_path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    # Transform polygons to image CRS
    to_img_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    polygons = []
    for feat in farmland["features"]:
        try:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)  # Fix invalid geometries
            geom_proj = shapely_transform(to_img_crs.transform, geom)
            if geom_proj.is_valid and not geom_proj.is_empty:
                polygons.append(geom_proj)
        except Exception:
            continue

    if polygons:
        mask = rasterize(
            [(g, 1) for g in polygons],
            out_shape=(height, width),
            transform=transform,
            fill=0, dtype=np.uint8,
        )
    else:
        print("  WARNING: No valid farmland polygons found!")
        mask = np.zeros((height, width), dtype=np.uint8)

    farm_pct = mask.sum() / mask.size * 100
    print(f"  Farmland pixels: {mask.sum():,} ({farm_pct:.1f}%)")

    out_path = os.path.join(output_dir, "farmland_mask.tif")
    with rasterio.open(out_path, "w", driver="GTiff", dtype="uint8",
                       width=width, height=height, count=1,
                       crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(mask, 1)

    print(f"  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
# 4. Tile into Patches
# ═══════════════════════════════════════════════════════════════

def tile_region(image_path, mask_path, out_dir, tile_size=256, stride=128,
                min_farm_ratio=0.001, keep_empty_prob=0.35):
    """Tile a region's image+mask into training patches."""
    from rasterio.windows import Window

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
        count = 0
        skipped = 0

        for y in range(0, img_src.height - tile_size + 1, stride):
            for x in range(0, img_src.width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)

                img_tile = img_src.read(window=window)  # (4, 256, 256)
                msk_tile = msk_src.read(1, window=window)  # (256, 256)

                # Skip tiles that are mostly nodata (black)
                if img_tile.max() == 0:
                    skipped += 1
                    continue

                # Filter by farmland content
                farm_ratio = msk_tile.sum() / msk_tile.size
                if farm_ratio < min_farm_ratio:
                    if np.random.rand() > keep_empty_prob:
                        skipped += 1
                        continue

                tile_transform = rasterio.windows.transform(window, img_src.transform)
                tile_name = f"tile_{count:05d}.tif"

                # Save image tile (4 bands)
                img_profile = img_src.profile.copy()
                img_profile.update(
                    width=tile_size, height=tile_size,
                    transform=tile_transform, driver="GTiff", compress="lzw",
                )
                with rasterio.open(
                    os.path.join(out_dir, "images", tile_name), "w", **img_profile
                ) as dst:
                    dst.write(img_tile)

                # Save mask tile
                msk_profile = msk_src.profile.copy()
                msk_profile.update(
                    width=tile_size, height=tile_size,
                    transform=tile_transform, driver="GTiff", compress="lzw",
                )
                with rasterio.open(
                    os.path.join(out_dir, "masks", tile_name), "w", **msk_profile
                ) as dst:
                    dst.write(msk_tile, 1)

                count += 1

    print(f"  Tiles: {count} saved, {skipped} skipped (low farm content / nodata)")
    return count


# ═══════════════════════════════════════════════════════════════
# 5. Processing & Merging
# ═══════════════════════════════════════════════════════════════

ALL_STEPS = ["download_sentinel", "download_osm", "rasterize", "tile"]


def _get_image_bbox_wgs84(image_path):
    """Get the actual WGS84 bounding box of a downloaded Sentinel-2 image."""
    from pyproj import Transformer
    import rasterio as rio

    with rio.open(image_path) as src:
        bounds = src.bounds
        img_crs = src.crs

    to_wgs84 = Transformer.from_crs(img_crs, "EPSG:4326", always_xy=True)
    west, south = to_wgs84.transform(bounds.left, bounds.bottom)
    east, north = to_wgs84.transform(bounds.right, bounds.top)
    return (west, south, east, north)


def process_region(region_key, base_dir, max_cloud, tile_size, stride, steps=None):
    """Download, rasterize, and tile data for one region."""
    if steps is None:
        steps = ALL_STEPS

    config = REGION_CONFIGS[region_key]
    bbox = config["bbox"]
    region_dir = os.path.join(base_dir, region_key)
    os.makedirs(region_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  {config['name']} — bbox: {bbox}")
    print(f"{'=' * 60}")

    image_path = os.path.join(region_dir, "sentinel2_rgbnir.tif")
    farmland_path = os.path.join(region_dir, "farmland.geojson")
    mask_path = os.path.join(region_dir, "farmland_mask.tif")

    # Step 1: Sentinel-2
    if "download_sentinel" in steps:
        result = download_sentinel2(region_key, bbox, region_dir, max_cloud)
        if result is None:
            return 0
        image_path = result

    # Step 2: OSM Farmland — use ACTUAL image bounds, not configured bbox
    if "download_osm" in steps:
        osm_bbox = bbox
        if os.path.exists(image_path):
            osm_bbox = _get_image_bbox_wgs84(image_path)
            print(f"  Using actual image bounds for OSM: {tuple(round(x,4) for x in osm_bbox)}")
        result = download_osm_farmland(region_key, osm_bbox, region_dir)
        if result is None:
            return 0
        farmland_path = result

    # Step 3: Rasterize
    if "rasterize" in steps:
        if not os.path.exists(image_path) or not os.path.exists(farmland_path):
            print("  ERROR: Need sentinel2_rgbnir.tif and farmland.geojson first.")
            return 0
        result = rasterize_farmland(region_key, image_path, farmland_path, region_dir)
        mask_path = result

    # Step 4: Tile
    if "tile" in steps:
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print("  ERROR: Need sentinel2_rgbnir.tif and farmland_mask.tif first.")
            return 0
        print(f"\n  [Tile] Cutting into {tile_size}x{tile_size} patches...")
        tile_dir = os.path.join(region_dir, "tiles")
        n_tiles = tile_region(image_path, mask_path, tile_dir, tile_size, stride)
        return n_tiles

    return 0


def merge_tiles(base_dir, regions, out_dir, val_split=0.15):
    """Merge tiled patches from all regions into train/val splits."""
    import shutil
    import random

    train_img = os.path.join(out_dir, "train", "images")
    train_msk = os.path.join(out_dir, "train", "masks")
    val_img = os.path.join(out_dir, "val", "images")
    val_msk = os.path.join(out_dir, "val", "masks")

    for d in [train_img, train_msk, val_img, val_msk]:
        os.makedirs(d, exist_ok=True)

    all_tiles = []
    for region_key in regions:
        tile_dir = os.path.join(base_dir, region_key, "tiles")
        img_dir = os.path.join(tile_dir, "images")
        msk_dir = os.path.join(tile_dir, "masks")

        if not os.path.isdir(img_dir):
            continue

        for name in os.listdir(img_dir):
            all_tiles.append((
                os.path.join(img_dir, name),
                os.path.join(msk_dir, name),
                region_key,
            ))

    if not all_tiles:
        print("WARNING: No tiles found to merge!")
        return

    random.shuffle(all_tiles)
    n_val = max(1, int(len(all_tiles) * val_split))

    print(f"\nMerging {len(all_tiles)} tiles -> {len(all_tiles) - n_val} train + {n_val} val")

    for i, (img_src, msk_src, region) in enumerate(all_tiles):
        new_name = f"{region}_{os.path.basename(img_src)}"

        if i < n_val:
            shutil.copy2(img_src, os.path.join(val_img, new_name))
            shutil.copy2(msk_src, os.path.join(val_msk, new_name))
        else:
            shutil.copy2(img_src, os.path.join(train_img, new_name))
            shutil.copy2(msk_src, os.path.join(train_msk, new_name))

    print(f"Train: {len(all_tiles) - n_val} tiles")
    print(f"Val:   {n_val} tiles")
    print(f"Saved to: {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 + OSM farmland for Indian agricultural regions",
        epilog="\n".join([
            "Examples:",
            "  python download_data.py --regions all",
            "  python download_data.py --regions punjab haryana",
            "  python download_data.py --bbox 72.50 22.10 72.70 22.30 --name anand_farms",
            "  python download_data.py --regions punjab --steps download_sentinel download_osm",
        ]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--regions", nargs="+", default=["all"],
                        help="Regions to download: " + ", ".join(REGION_CONFIGS.keys()) + ", or 'all'")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Custom bounding box (WGS84). Use bboxfinder.com to get coordinates.")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for custom bbox area (required with --bbox)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base data directory")
    parser.add_argument("--max_cloud", type=int, default=15,
                        help="Max cloud cover (%%)")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--steps", nargs="+", default=None,
                        choices=ALL_STEPS,
                        help="Run specific steps only (default: all steps)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, just merge existing tiles")
    args = parser.parse_args()

    # Handle custom bbox
    if args.bbox:
        if not args.name:
            print("ERROR: --name is required when using --bbox")
            print("Example: --bbox 72.50 22.10 72.70 22.30 --name anand_farms")
            sys.exit(1)

        custom_key = args.name.lower().replace(" ", "_")
        bbox = tuple(args.bbox)
        center_lon = (bbox[0] + bbox[2]) / 2

        REGION_CONFIGS[custom_key] = {
            "name": args.name,
            "bbox": bbox,
            "utm_crs": get_utm_crs(center_lon),
        }
        print(f"Custom area registered: {args.name}")
        print(f"  bbox: {bbox}")
        print(f"  UTM CRS: {REGION_CONFIGS[custom_key]['utm_crs']}")

    # Resolve region list
    if args.bbox and "all" not in args.regions:
        regions = [r.lower() for r in args.regions if r != "all"]
        custom_key = args.name.lower().replace(" ", "_")
        if custom_key not in regions:
            regions.append(custom_key)
    elif args.bbox and "all" in args.regions and len(args.regions) == 1:
        custom_key = args.name.lower().replace(" ", "_")
        regions = [custom_key]
    elif "all" in args.regions:
        regions = list(REGION_CONFIGS.keys())
    else:
        regions = [r.lower() for r in args.regions]

    for r in regions:
        if r not in REGION_CONFIGS:
            print(f"Unknown region: {r}. Available: {list(REGION_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Regions: {', '.join(REGION_CONFIGS[r]['name'] for r in regions)}")
    print(f"Output: {args.data_dir}/")

    # Process each region
    if not args.skip_download:
        total_tiles = 0
        for region_key in regions:
            try:
                n = process_region(
                    region_key, args.data_dir, args.max_cloud,
                    args.tile_size, args.stride, args.steps,
                )
                total_tiles += n
            except Exception as e:
                print(f"\n  ERROR processing {region_key}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Brief pause between regions to be nice to APIs
            time.sleep(2)

        print(f"\n{'=' * 60}")
        print(f"Total tiles across all regions: {total_tiles}")
        print(f"{'=' * 60}")

    # Merge all region tiles into train/val
    merged_dir = os.path.join(args.data_dir, "combined")
    merge_tiles(args.data_dir, regions, merged_dir)

    print(f"\n{'=' * 60}")
    print(f"DONE! Ready to train:")
    print(f"  python train.py --data_dir {merged_dir} --epochs 50")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
