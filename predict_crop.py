import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# --- 1. Lightweight U-Net Model (must match training code) ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        d3 = self.up3(e4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

def main():
    parser = argparse.ArgumentParser(description="Test Custom Crop U-Net")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--weights", type=str, default="checkpoints_unet/best_fast_unet.pth")
    parser.add_argument("--output", type=str, default="predictions_crop")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = SimpleUNet().to(device)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
        
    model.eval()

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # Transform (matches training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    orig_img = Image.open(args.image).convert("RGB")
    orig_size = orig_img.size # (width, height)
    
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Apply sigmoid to convert logits to probabilities
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
    # Threshold at a balanced 0.5
    mask = (prob > 0.5).astype(np.uint8) * 255
    
    # Resize mask back to match original image size for clean overlay
    mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    
    # --- Watershed Post-Processing to Fix "Blobbing" ---
    # 1. Cleaning up the mask using morph open (remove noise) then close (fill tiny holes)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 2. Finding sure foreground area using Distance Transform
    # This finds the exact optical centers of the fields and drops the touching blurry edges
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 4. Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # Add one to all labels so that background is not 0, but 1
    markers[unknown == 255] = 0 # Mark the unknown region with 0
    
    # 5. Apply Watershed
    # Convert orig_img to BGR for OpenCV watershed
    img_bgr = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)
    
    # Create the final split mask (anything > 1 is a distinct field)
    mask_separated = np.zeros_like(mask_resized)
    mask_separated[markers > 1] = 255
    
    # Extract boundaries to draw outlines
    contours, _ = cv2.findContours(mask_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    orig_np = np.array(orig_img)
    # Draw green outlines around the separated fields
    cv2.drawContours(orig_np, contours, -1, (0, 255, 0), 2)
    
    # Add a slight green transparent overlay over the filled areas
    color_mask = np.zeros_like(orig_np)
    color_mask[:, :, 1] = mask_separated
    alpha = 0.3
    overlay = cv2.addWeighted(orig_np, 1, color_mask, alpha, 0)
    
    out_filename = os.path.join(args.output, "pred_" + os.path.basename(args.image))
    out_img = Image.fromarray(overlay)
    out_img.save(out_filename)
    
    print(f"Prediction saved to: {out_filename}")

if __name__ == "__main__":
    main()
