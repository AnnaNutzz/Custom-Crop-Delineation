"""
small_area_demo.py — Quick demo: download + predict on a small area
====================================================================
Downloads Sentinel-2 imagery for a small (~5 km²) agricultural area
near Anand, Gujarat, runs the trained U-Net model, and produces
publication-quality visualizations.

Usage:
    python small_area_demo.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box, mapping
import json
import requests

# ─── Configuration ───────────────────────────────────────────────
# Small agricultural area near Anand, Gujarat (~5 km × 5 km)
DEMO_BBOX = (72.93, 22.53, 72.98, 22.58)   # (west, south, east, north)
AREA_NAME = "Anand, Gujarat"
MODEL_PATH = "checkpoints/best_model.pth"
OUTPUT_DIR = "small_area_results"
MAX_CLOUD = 20
THRESHOLD = 0.45

# ═══════════════════════════════════════════════════════════════
# 1. Download Sentinel-2
# ═══════════════════════════════════════════════════════════════

def download_sentinel2(bbox, output_dir):
    """Download Sentinel-2 RGB+NIR for the demo area."""
    from pystac_client import Client
    import planetary_computer
    from pyproj import Transformer
    from rasterio.mask import mask as rio_mask

    print(f"\n📡 Downloading Sentinel-2 for {AREA_NAME}...")
    print(f"   bbox: {bbox}")

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        sortby=["-properties.datetime"],
        max_items=5,
    )

    items = list(search.items())
    if not items:
        print("   ERROR: No scenes found! Try increasing MAX_CLOUD.")
        sys.exit(1)

    item = items[0]
    scene_date = item.properties["datetime"][:10]
    cloud_pct = item.properties["eo:cloud_cover"]
    print(f"   Scene: {item.id}")
    print(f"   Date:  {scene_date}")
    print(f"   Cloud: {cloud_pct:.1f}%")

    band_names = ["B04", "B03", "B02", "B08"]  # R, G, B, NIR
    bands = []
    out_transform = out_crs = profile = None

    for band_name in band_names:
        asset = item.assets[band_name]
        print(f"   Downloading {band_name}...", end=" ", flush=True)
        with rasterio.open(asset.href) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            west, south = transformer.transform(bbox[0], bbox[1])
            east, north = transformer.transform(bbox[2], bbox[3])
            aoi_utm = box(west, south, east, north)
            data, transform = rio_mask(src, [mapping(aoi_utm)], crop=True)
            bands.append(data[0])
            out_transform = transform
            out_crs = src.crs
            profile = src.profile.copy()
            print(f"OK ({data.shape[1]}×{data.shape[2]})")

    stacked = np.stack(bands, axis=0)
    h, w = stacked.shape[1], stacked.shape[2]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sentinel2_rgbnir.tif")
    profile.update(
        driver="GTiff", count=4, height=h, width=w,
        transform=out_transform, crs=out_crs, compress="lzw",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    print(f"   ✅ Saved: {out_path} ({w}×{h} px, 4 bands)")
    return out_path, scene_date


# ═══════════════════════════════════════════════════════════════
# 2. Download OSM Farmland
# ═══════════════════════════════════════════════════════════════

def download_osm(bbox, output_dir):
    """Download OSM farmland polygons for comparison."""
    print(f"\n🗺️  Downloading OSM farmland polygons...")

    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    query = f"""[out:json][timeout:120];
(
  way["landuse"="farmland"]({south},{west},{north},{east});
  way["landuse"="farm"]({south},{west},{north},{east});
  way["landuse"="orchard"]({south},{west},{north},{east});
  relation["landuse"="farmland"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query}, timeout=120,
    )

    if response.status_code != 200:
        print(f"   WARNING: Overpass returned {response.status_code}, skipping OSM.")
        return None

    data = response.json()
    nodes = {}
    raw_ways = []

    for el in data["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el["type"] == "way":
            raw_ways.append(el)

    features = []
    for el in raw_ways:
        coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
        if len(coords) >= 4:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": el.get("tags", {}),
            })

    out_path = os.path.join(output_dir, "farmland.geojson")
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    print(f"   ✅ Found {len(features)} farmland polygons")
    return out_path


def rasterize_osm(image_path, geojson_path, output_dir):
    """Rasterize OSM polygons to match Sentinel-2 grid."""
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    with open(geojson_path) as f:
        farmland = json.load(f)

    with rasterio.open(image_path) as src:
        crs = src.crs
        transform = src.transform
        w, h = src.width, src.height

    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    polygons = []
    for feat in farmland["features"]:
        try:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            geom_proj = shapely_transform(to_crs.transform, geom)
            if geom_proj.is_valid and not geom_proj.is_empty:
                polygons.append(geom_proj)
        except Exception:
            continue

    if polygons:
        mask = rasterize(
            [(g, 1) for g in polygons],
            out_shape=(h, w), transform=transform,
            fill=0, dtype=np.uint8,
        )
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    pct = mask.sum() / mask.size * 100
    print(f"   OSM farmland coverage: {pct:.1f}%")
    return mask


# ═══════════════════════════════════════════════════════════════
# 3. Load Model & Predict
# ═══════════════════════════════════════════════════════════════

def load_model(model_path, device):
    """Load trained U-Net model."""
    import segmentation_models_pytorch as smp

    print(f"\n🧠 Loading model from: {model_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in state:
        encoder_name = state.get("encoder_name", "efficientnet-b0")
        in_channels = state.get("in_channels", 4)
        best_iou = state.get("best_iou", 0)
        state_dict = state["model_state_dict"]
        print(f"   Encoder:  {encoder_name}")
        print(f"   Channels: {in_channels}")
        print(f"   Best IoU: {best_iou:.4f}")
    else:
        encoder_name = "efficientnet-b0"
        in_channels = 4
        best_iou = 0
        state_dict = state

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        decoder_attention_type="scse",
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"   ✅ Model loaded on {device}")
    return model, in_channels, best_iou


def predict_large(model, image, device, in_channels=4,
                  tile_size=256, overlap=64, threshold=0.5):
    """Sliding-window prediction for large images."""
    c, h, w = image.shape
    if c > in_channels:
        image = image[:in_channels]
    elif c < in_channels:
        pad = np.zeros((in_channels - c, h, w), dtype=image.dtype)
        image = np.concatenate([image, pad], axis=0)

    stride = tile_size - overlap
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    total = ((h - tile_size) // stride + 1) * ((w - tile_size) // stride + 1)
    done = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size
            if y_start < 0 or x_start < 0:
                continue

            tile = image[:, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                t = torch.from_numpy(tile).unsqueeze(0).float().to(device)
                prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()

            prob_map[y_start:y_end, x_start:x_end] += prob
            count_map[y_start:y_end, x_start:x_end] += 1
            done += 1

        print(f"\r   Predicting... {done}/{total} tiles", end="", flush=True)

    print()
    count_map = np.maximum(count_map, 1)
    prob_map /= count_map
    mask = (prob_map > threshold).astype(np.uint8)
    return prob_map, mask


# ═══════════════════════════════════════════════════════════════
# 4. Visualization
# ═══════════════════════════════════════════════════════════════

def create_report(image, prob_map, pred_mask, osm_mask, scene_date,
                  best_iou, output_dir):
    """Create a publication-quality multi-panel report image."""
    print(f"\n🎨 Creating visualization report...")

    # Prepare RGB
    rgb = image[:3].transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)

    # Brightness boost for visibility
    p2, p98 = np.percentile(rgb, [2, 98])
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

    # NDVI
    nir = image[3]
    red = image[0]
    ndvi = (nir - red) / (nir + red + 1e-7)
    ndvi = np.clip(ndvi, -1, 1)

    # Stats
    pred_pct = pred_mask.sum() / pred_mask.size * 100
    osm_pct = osm_mask.sum() / osm_mask.size * 100 if osm_mask is not None else 0
    h, w = pred_mask.shape
    area_km2 = (DEMO_BBOX[2] - DEMO_BBOX[0]) * (DEMO_BBOX[3] - DEMO_BBOX[1]) * 111 * 111 * 0.85

    # ─── Figure ───
    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15,
                  left=0.03, right=0.97, top=0.88, bottom=0.08)

    title_color = "#e0e0e0"
    panel_bg = "#16213e"

    # Title
    fig.suptitle(
        f"Crop Field Delineation — {AREA_NAME}",
        fontsize=22, fontweight="bold", color="white", y=0.96
    )
    fig.text(0.5, 0.92,
             f"Sentinel-2 ({scene_date})  •  {w}×{h} px  •  ~{area_km2:.1f} km²  •  Model IoU: {best_iou:.3f}",
             ha="center", fontsize=12, color="#aaaaaa")

    # Panel 1: RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title("Sentinel-2 RGB", fontsize=14, color=title_color, pad=10)
    ax1.axis("off")

    # Panel 2: NDVI
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax2.set_title("NDVI (Vegetation Index)", fontsize=14, color=title_color, pad=10)
    ax2.axis("off")
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors=title_color, labelsize=9)

    # Panel 3: Probability Map
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(prob_map, cmap="YlGn", vmin=0, vmax=1)
    ax3.set_title("Field Probability Map", fontsize=14, color=title_color, pad=10)
    ax3.axis("off")
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors=title_color, labelsize=9)

    # Panel 4: Predicted Mask Overlay
    ax4 = fig.add_subplot(gs[1, 0])
    overlay = rgb.copy()
    fm = pred_mask.astype(bool)
    overlay[fm] = overlay[fm] * 0.35 + np.array([0.1, 0.85, 0.2]) * 0.65
    ax4.imshow(overlay)
    ax4.set_title(f"Predicted Fields ({pred_pct:.1f}% coverage)", fontsize=14,
                  color=title_color, pad=10)
    ax4.axis("off")

    # Panel 5: OSM Ground Truth (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    if osm_mask is not None and osm_mask.sum() > 0:
        overlay_osm = rgb.copy()
        om = osm_mask.astype(bool)
        overlay_osm[om] = overlay_osm[om] * 0.35 + np.array([0.9, 0.6, 0.1]) * 0.65
        ax5.imshow(overlay_osm)
        ax5.set_title(f"OSM Farmland ({osm_pct:.1f}% coverage)", fontsize=14,
                      color=title_color, pad=10)
    else:
        ax5.imshow(rgb)
        ax5.set_title("OSM Farmland (unavailable)", fontsize=14,
                      color=title_color, pad=10)
    ax5.axis("off")

    # Panel 6: Comparison Overlay (Pred vs OSM)
    ax6 = fig.add_subplot(gs[1, 2])
    comparison = rgb.copy()
    if osm_mask is not None and osm_mask.sum() > 0:
        both = fm & om
        pred_only = fm & ~om
        osm_only = ~fm & om
        comparison[both] = comparison[both] * 0.3 + np.array([0.1, 0.85, 0.2]) * 0.7
        comparison[pred_only] = comparison[pred_only] * 0.3 + np.array([0.2, 0.5, 1.0]) * 0.7
        comparison[osm_only] = comparison[osm_only] * 0.3 + np.array([1.0, 0.4, 0.1]) * 0.7
        ax6.set_title("Comparison: 🟢 Both  🔵 Model Only  🟠 OSM Only",
                      fontsize=12, color=title_color, pad=10)
    else:
        comparison[fm] = comparison[fm] * 0.35 + np.array([0.1, 0.85, 0.2]) * 0.65
        ax6.set_title(f"Detected Crop Fields", fontsize=14,
                      color=title_color, pad=10)
    ax6.imshow(comparison)
    ax6.axis("off")

    # Footer
    fig.text(0.5, 0.02,
             f"U-Net (EfficientNet-B0) • Threshold: {THRESHOLD} • "
             f"bbox: [{DEMO_BBOX[0]:.2f}, {DEMO_BBOX[1]:.2f}, {DEMO_BBOX[2]:.2f}, {DEMO_BBOX[3]:.2f}]",
             ha="center", fontsize=10, color="#777777")

    report_path = os.path.join(output_dir, "report_results.png")
    fig.savefig(report_path, dpi=180, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Report saved: {report_path}")

    # Also save a simple overlay for quick sharing
    fig2, ax = plt.subplots(1, 1, figsize=(10, 10))
    overlay_simple = rgb.copy()
    overlay_simple[fm] = overlay_simple[fm] * 0.35 + np.array([0.1, 0.85, 0.2]) * 0.65
    ax.imshow(overlay_simple)
    ax.set_title(f"Crop Field Detection — {AREA_NAME}\n"
                 f"Sentinel-2 ({scene_date}) • {pred_pct:.1f}% field coverage",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    simple_path = os.path.join(output_dir, "field_overlay.png")
    fig2.savefig(simple_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"   ✅ Simple overlay saved: {simple_path}")

    return report_path, simple_path


# ═══════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f" CROP FIELD DELINEATION — SMALL AREA DEMO")
    print(f" Area: {AREA_NAME}")
    print(f" bbox: {DEMO_BBOX}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Download imagery
    image_path, scene_date = download_sentinel2(DEMO_BBOX, OUTPUT_DIR)

    # Step 2: Load image
    print(f"\n📂 Loading image...")
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    if image.max() > 1.0:
        if image.max() > 255:
            image = np.clip(image / 10000.0, 0.0, 1.0)
        else:
            image = image / 255.0

    c, h, w = image.shape
    print(f"   Image: {w}×{h} px, {c} bands")

    # Step 3: Download OSM ground truth
    osm_mask = None
    geojson_path = download_osm(DEMO_BBOX, OUTPUT_DIR)
    if geojson_path:
        osm_mask = rasterize_osm(image_path, geojson_path, OUTPUT_DIR)

    # Step 4: Load model & predict
    model, in_channels, best_iou = load_model(MODEL_PATH, device)

    print(f"\n🔮 Running prediction on {w}×{h} image...")
    prob_map, pred_mask = predict_large(
        model, image, device, in_channels,
        tile_size=256, overlap=64, threshold=THRESHOLD
    )

    pred_pct = pred_mask.sum() / pred_mask.size * 100
    print(f"   ✅ Field coverage: {pred_pct:.1f}%")

    # Step 5: Save mask GeoTIFF
    mask_path = os.path.join(OUTPUT_DIR, "predicted_field_mask.tif")
    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8", compress="lzw")
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write((pred_mask * 255).astype(np.uint8), 1)
    print(f"   ✅ Mask GeoTIFF: {mask_path}")

    # Step 6: Create visualizations
    report_path, simple_path = create_report(
        image, prob_map, pred_mask, osm_mask, scene_date, best_iou, OUTPUT_DIR
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f" ✅ DONE — Results in: {OUTPUT_DIR}/")
    print(f"    • report_results.png  — Full 6-panel report")
    print(f"    • field_overlay.png   — Simple overlay (easy to share)")
    print(f"    • predicted_field_mask.tif  — GeoTIFF mask")
    print(f"    • sentinel2_rgbnir.tif     — Raw imagery")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
