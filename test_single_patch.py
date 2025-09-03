import torch
import numpy as np
import tifffile
from model_hpc import load_trained_model
from dataset_hpc import fixed_normalise
import os


def test_single_patch(model_path, config_path, volume_path, x,y,z):
    """Test model on a single 128^3 patch"""

    # Load model
    model = load_trained_model(model_path, config_path)
    model.eval()

    # Load volume
    if os.path.isdir(volume_path):
        # Load first slice as example
        files = sorted([f for f in os.listdir(volume_path) if f.endswith('.tif')])
        volume = np.stack([tifffile.imread(os.path.join(volume_path, f)) for f in files[z:z+128]])
    else:
        volume = tifffile.imread(volume_path)

    # Normalize
    #volume = fixed_normalise(volume, -0.01, 0.025)

    # Extract single patch
    patch = volume[:, y:y + 128, x:x + 128]

    # Convert to tensor
    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    # Run inference
    with torch.no_grad():
        logits = model(patch_tensor)
        probs = torch.sigmoid(logits)

    # Extract results
    foreground = probs[0, 0].cpu().numpy()
    boundary = probs[0, 1].cpu().numpy()

    # Save results
    tifffile.imwrite(f'test/single_patch_input{z}_{y}_{x}.tif', patch)
    tifffile.imwrite(f'test/single_patch_foreground{z}_{y}_{x}.tif', foreground)
    tifffile.imwrite(f'test/single_patch_boundary{z}_{y}_{x}.tif', boundary)

    print(f"Logit range: {logits.min().item():.2f} to {logits.max().item():.2f}")
    print(f"Probability range: {probs.min().item():.4f} to {probs.max().item():.4f}")
    print(f"Foreground pixels: {np.sum(foreground > 0.5)} / {foreground.size}")

    return patch, foreground, boundary


# Usage
model_path = "/mnt/duke-netapp/asvetlove/plankton_results/job_36471054/checkpoints/best_model.pth"
config_path = "/mnt/duke-netapp/asvetlove/plankton_results/job_36471054/logs/config.json"
volume_path = "/home/asvetlove/data/segmentation/inference_examples/POR_20to200_20231022_AM_01_epo_02/"

# Test center patch
patch_coords = (200, 1000, 1000)  # Pick coordinates with plankton
x = [1355,
731,
1724,
2765,
986,
881,
1562,
1451,
1178,
350,
1076]
y = [964,
718,
1873,
1192,
1981,
2239,
2506,
2068,
2851,
2554,
406]
z=[324,
324,
324,
324,
324,
324,
324,
324,
324,
324,
324]
for _x,_y,_z in zip(x, y, z):
    test_single_patch(model_path, config_path, volume_path, _x,_y,_z)