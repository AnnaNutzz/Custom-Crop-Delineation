import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time
import numpy as np

# --- 1. Lightweight U-Net Model ---
class SimpleUNet(nn.Module):
    """A lighter, faster U-Net tailored for binary segmentation on a 4GB VRAM GPU."""
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

# --- 2. Dataset Loader ---
class CropDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Map only the images that have a corresponding mask
        all_masks = set(f for f in os.listdir(mask_dir) if f.endswith('.png'))
        self.images = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith('.jpg') and f.replace('.jpg', '.png') in all_masks
        ])
        
        self.masks = sorted([
            f.replace('.jpg', '.png') for f in self.images
        ])
        
        # Basic train transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)), # Downsize heavily for very fast training
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Match naming convention from json_to_mask (india_tile_001.jpg -> india_tile_001.png)
        mask_name = img_name.replace('.jpg', '.png')
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = self.transform(image)
        mask = self.mask_transform(mask)
        
        # Binarize mask exactly (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask

# --- 3. Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Fast U-Net Training for Custom Crop dataset")
    parser.add_argument("--data_dir", type=str, default="india_dataset", help="Root directory containing images/ and masks/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "images")
    mask_dir = os.path.join(args.data_dir, "masks")
    
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        print(f"Directory {img_dir} or {mask_dir} not found. Annotate data first!")
        return

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check dataset
    dataset = CropDataset(img_dir, mask_dir)
    print(f"Found {len(dataset)} labeled images.")
    if len(dataset) == 0:
        print("No valid image/mask pairs found. Exiting.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Initialize model, criterion, optimizer, and scaler for fast Mixed Precision
    model = SimpleUNet().to(device)
    # Using BCEWithLogitsLoss for binary segmentation (faster and more stable)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # AMP Scaler for huge speed boost on modern GPUs
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("checkpoints_unet", exist_ok=True)
    best_loss = float("inf")

    print(f"\n[START] Training fast U-Net for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # --- Mixed Precision Forward Pass (Major Speedup) ---
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            # --- Mixed Precision Backward Pass ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] -> Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save dict
            torch.save(model.state_dict(), "checkpoints_unet/best_fast_unet.pth")
            print("  -> Saved new best model")

    print(f"\n[DONE] Fast Training Complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
