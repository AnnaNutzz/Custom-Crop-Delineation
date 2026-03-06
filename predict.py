"""
predict.py — Run crop field detection on satellite imagery
============================================================
Loads a trained model and predicts farmland masks for input images.

Usage:
    # Predict on a single GeoTIFF:
    python predict.py --model checkpoints/best_model.pth --input data/punjab/sentinel2_rgbnir.tif

    # Predict on a folder of tiles:
    python predict.py --model checkpoints/best_model.pth --input data/combined/val/images/

    # Adjust threshold (lower = more fields detected):
    python predict.py --model checkpoints/best_model.pth --input image.tif --threshold 0.4

    # RGB-only mode (3-channel input):
    python predict.py --model checkpoints/best_model.pth --input image.tif --in_channels 3
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


# ═══════════════════════════════════════════════════════════════
# 1. Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model(model_path, device, in_channels=None):
    """Load trained U-Net model from checkpoint. Auto-detects encoder and channels."""
    print(f"Loading model from: {model_path}")

    state = torch.load(model_path, map_location="cpu", weights_only=False)

    # Extract config from checkpoint
    if "model_state_dict" in state:
        encoder_name = state.get("encoder_name", "efficientnet-b0")
        saved_channels = state.get("in_channels", 4)
        best_iou = state.get("best_iou", 0)
        state_dict = state["model_state_dict"]
        print(f"  Encoder: {encoder_name}")
        print(f"  Channels: {saved_channels}")
        print(f"  Best IoU: {best_iou:.4f}")
    else:
        # Raw state dict — try to detect encoder from key names
        encoder_name = "efficientnet-b0"
        saved_channels = in_channels or 4
        state_dict = state
        for key in state_dict:
            if "resnet" in key.lower() or key.startswith("encoder.conv1"):
                encoder_name = "resnet34"
                break
        print(f"  Encoder (guessed): {encoder_name}")

    model_channels = in_channels or saved_channels

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=model_channels,
        classes=1,
        decoder_attention_type="scse",
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("  WARNING: Loaded with strict=False (partial match)")

    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully on {device}")
    return model, model_channels


# ═══════════════════════════════════════════════════════════════
# 2. Prediction Functions
# ═══════════════════════════════════════════════════════════════

def predict_tile(model, image_np, device, threshold=0.5):
    """
    Predict crop field mask for a single image tile.

    Args:
        model: trained model
        image_np: numpy array (C, H, W), float32 [0, 1]
        device: torch device
        threshold: probability threshold for binary mask

    Returns:
        pred_prob: (H, W) float probability map
        pred_mask: (H, W) binary mask
    """
    with torch.no_grad():
        x = torch.from_numpy(image_np).unsqueeze(0).float().to(device)
        out = torch.sigmoid(model(x))
        prob = out[0, 0].cpu().numpy()
        mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def predict_large_image(model, image, device, in_channels=4,
                        tile_size=256, overlap=64, threshold=0.5):
    """
    Predict crop field mask for a large image by sliding window.

    Args:
        model: trained model
        image: (C, H, W) numpy array
        device: torch device
        in_channels: number of input channels
        tile_size: prediction tile size
        overlap: overlap between tiles
        threshold: probability threshold

    Returns:
        prob_map: (H, W) float probability map
        mask: (H, W) binary mask
    """
    c, h, w = image.shape

    # Handle channel mismatch
    if c > in_channels:
        image = image[:in_channels]
        c = in_channels
    elif c < in_channels:
        pad = np.zeros((in_channels - c, h, w), dtype=image.dtype)
        image = np.concatenate([image, pad], axis=0)
        c = in_channels

    stride = tile_size - overlap
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size

            if y_start < 0 or x_start < 0:
                continue

            tile = image[:, y_start:y_end, x_start:x_end]
            prob, _ = predict_tile(model, tile, device, threshold)

            prob_map[y_start:y_end, x_start:x_end] += prob
            count_map[y_start:y_end, x_start:x_end] += 1

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1)
    prob_map /= count_map
    mask = (prob_map > threshold).astype(np.uint8)

    return prob_map, mask


# ═══════════════════════════════════════════════════════════════
# 3. I/O Functions
# ═══════════════════════════════════════════════════════════════

def load_image(path, in_channels=4):
    """Load image from GeoTIFF or PNG."""
    import rasterio

    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
            profile = src.profile.copy()
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        image = np.array(img, dtype=np.float32).transpose(2, 0, 1)
        profile = None

    # Normalize to [0, 1]
    if image.max() > 1.0:
        if image.max() > 255:
            image = np.clip(image / 10000.0, 0.0, 1.0)
        else:
            image = image / 255.0

    # Handle channel count
    if image.shape[0] > in_channels:
        image = image[:in_channels]
    elif image.shape[0] < in_channels:
        pad = np.zeros((in_channels - image.shape[0],) + image.shape[1:],
                       dtype=image.dtype)
        image = np.concatenate([image, pad], axis=0)

    return image, profile


def save_mask_geotiff(mask, profile, output_path):
    """Save predicted mask as GeoTIFF (0=background, 255=field)."""
    import rasterio

    if profile is None:
        return

    profile.update(count=1, dtype="uint8", compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write((mask * 255).astype(np.uint8), 1)


# ═══════════════════════════════════════════════════════════════
# 4. Visualization
# ═══════════════════════════════════════════════════════════════

def visualize_result(image, prob_map, mask, save_path, in_channels=4,
                     title="Crop Field Detection"):
    """Create a multi-panel visualization."""
    n_cols = 4 if in_channels >= 4 else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    col = 0

    # Panel 1: RGB
    rgb = image[:3].transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)
    axes[col].imshow(rgb)
    axes[col].set_title("Input RGB")
    axes[col].axis("off")
    col += 1

    # Panel 2: NDVI (if 4-band)
    if in_channels >= 4 and image.shape[0] >= 4:
        nir = image[3]
        red = image[0]
        ndvi = (nir - red) / (nir + red + 1e-7)
        ndvi = np.clip(ndvi, -1, 1)
        im = axes[col].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        axes[col].set_title("NDVI")
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046)
        col += 1

    # Panel 3: Probability map
    im = axes[col].imshow(prob_map, cmap="YlGn", vmin=0, vmax=1)
    axes[col].set_title("Field Probability")
    axes[col].axis("off")
    plt.colorbar(im, ax=axes[col], fraction=0.046)
    col += 1

    # Panel 4: Overlay
    overlay = rgb.copy()
    field_mask = mask.astype(bool)
    overlay[field_mask] = overlay[field_mask] * 0.4 + np.array([0, 0.8, 0]) * 0.6
    axes[col].imshow(overlay)
    field_pct = mask.sum() / mask.size * 100
    axes[col].set_title(f"Detected Fields ({field_pct:.1f}%)")
    axes[col].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Visualization: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 5. Processing
# ═══════════════════════════════════════════════════════════════

def process_single_image(model, image_path, output_dir, device,
                         in_channels, tile_size, threshold):
    """Process one satellite image and save results."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing: {basename}")

    image, profile = load_image(image_path, in_channels)
    c, h, w = image.shape
    print(f"  Size: {w}x{h}, {c} bands")

    # Use sliding window for large images, direct prediction for tiles
    if h > tile_size * 1.5 or w > tile_size * 1.5:
        print(f"  Using sliding window (tile={tile_size}, overlap=64)...")
        prob_map, mask = predict_large_image(
            model, image, device, in_channels, tile_size, threshold=threshold)
    else:
        prob_map, mask = predict_tile(model, image, device, threshold)

    field_pct = mask.sum() / mask.size * 100
    print(f"  Field coverage: {field_pct:.1f}%")

    os.makedirs(output_dir, exist_ok=True)

    # Save mask as GeoTIFF
    mask_path = os.path.join(output_dir, f"{basename}_field_mask.tif")
    save_mask_geotiff(mask, profile, mask_path)
    print(f"  Mask: {mask_path}")

    # Save visualization
    viz_path = os.path.join(output_dir, f"{basename}_visualization.png")
    visualize_result(image, prob_map, mask, viz_path, in_channels,
                     title=f"Crop Field Detection — {basename}")

    return field_pct


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Predict crop field masks from satellite imagery",
        epilog="\n".join([
            "Examples:",
            "  python predict.py --model checkpoints/best_model.pth --input data/punjab/sentinel2_rgbnir.tif",
            "  python predict.py --model checkpoints/best_model.pth --input data/combined/val/images/",
            "  python predict.py --model checkpoints/best_model.pth --input image.tif --threshold 0.4",
        ]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pth)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image path or directory of tiles")
    parser.add_argument("--output", type=str, default="predictions",
                        help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for field detection")
    parser.add_argument("--tile_size", type=int, default=256,
                        help="Tile size for sliding window")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Override input channels (auto-detected from checkpoint)")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, model_channels = load_model(args.model, device, args.in_channels)
    in_channels = args.in_channels or model_channels

    # Collect input files
    if os.path.isdir(args.input):
        exts = {".tif", ".tiff", ".png", ".jpg"}
        files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"\nFound {len(files)} images in {args.input}")
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    # Process
    results = []
    for f in files:
        try:
            pct = process_single_image(
                model, f, args.output, device,
                in_channels, args.tile_size, args.threshold)
            results.append((os.path.basename(f), pct))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print(f"RESULTS SUMMARY")
        print(f"{'=' * 60}")
        for name, pct in results:
            print(f"  {name}: {pct:.1f}% field coverage")
        avg = np.mean([p for _, p in results])
        print(f"\n  Average field coverage: {avg:.1f}%")
        print(f"  Output: {args.output}/")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
