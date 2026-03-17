import os
import requests
from io import BytesIO
from PIL import Image

def download_xyz_tile(url, x, y, z, save_path):
    tile_url = url.format(x=x, y=y, z=z)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(tile_url, headers=headers)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            print(f"Saved {save_path}")
        else:
            print(f"Failed to download tile: {response.status_code}")
    except Exception as e:
        print(f"Error downloading tile: {e}")

URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# Define a function to grab a grid of tiles around a central point
def generate_tile_grid(center_x, center_y, z, grid_size=5):
    tiles = []
    # e.g. grid_size 5 means a 5x5 grid (25 tiles) around the center
    offset = grid_size // 2
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            tiles.append((center_x + dx, center_y + dy, z))
    return tiles

# Generate ~100 tiles by taking 5x5 grids across 4 different Indian agricultural regions
TILES_TO_DOWNLOAD = []
TILES_TO_DOWNLOAD.extend(generate_tile_grid(46505, 27690, 16)) # Punjab Region (25 tiles)
TILES_TO_DOWNLOAD.extend(generate_tile_grid(46800, 28100, 16)) # MP Region (25 tiles)
TILES_TO_DOWNLOAD.extend(generate_tile_grid(46300, 28500, 16)) # Gujarat Region (25 tiles)
TILES_TO_DOWNLOAD.extend(generate_tile_grid(47100, 28800, 16)) # Maharashtra Region (25 tiles)

def main():
    output_dir = "india_dataset/images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("india_dataset/labels_json", exist_ok=True)
    os.makedirs("india_dataset/masks", exist_ok=True)
    
    print(f"Downloading {len(TILES_TO_DOWNLOAD)} sample high-res tiles from India...")
    for i, (x, y, z) in enumerate(TILES_TO_DOWNLOAD):
        filename = os.path.join(output_dir, f"india_tile_{i+1:03d}.jpg")
        download_xyz_tile(URL, x, y, z, filename)
        
    print("\nDownload complete.")

if __name__ == "__main__":
    main()
