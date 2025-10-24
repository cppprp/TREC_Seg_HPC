import torch
import numpy as np
import tifffile
from model_hpc import load_trained_model
from dataset_hpc import fixed_normalise
import os
from matplotlib import pyplot as plt


def create_realistic_synthetic_volume(shape=(128, 128, 128)):
    # Use your real data range directly
    volume = np.random.normal(-0.005, 0.003, shape).astype(np.float32)  # Background

    # Add texture variations
    for _ in range(5):
        center = [np.random.randint(0, s) for s in shape]
        size = [np.random.randint(20, 40) for _ in range(3)]
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        blob_mask = ((z - center[0]) / size[0]) ** 2 + ((y - center[1]) / size[1]) ** 2 + (
                    (x - center[2]) / size[2]) ** 2 < 1
        volume[blob_mask] += np.random.normal(0.002, 0.001)

    # Create bright cube in your intensity range
    cube_center = [64, 64, 64]
    cube_size = 20
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cube_mask = ((np.abs(z - cube_center[0]) < cube_size / 2) &
                 (np.abs(y - cube_center[1]) < cube_size / 2) &
                 (np.abs(x - cube_center[2]) < cube_size / 2))

    # Bright cube in your range
    volume[cube_mask] = np.random.normal(0.020, 0.002, np.sum(cube_mask))

    label = cube_mask.astype(np.uint8)

    # Now normalization should work properly
    volume = fixed_normalise(volume, -0.01, 0.025)

    return volume, label

# Test with different scenarios
def test_feature_discrimination():
    """Test if model can distinguish different synthetic features"""

    shape = (128, 128, 128)

    # Test 1: High contrast cube (should be easy)
    vol1, lbl1 = create_realistic_synthetic_volume()

    # Test 2: Low contrast cube (harder)
    vol2, lbl2 = create_realistic_synthetic_volume()
    vol2[lbl2 > 0] *= 0.7  # Make cube dimmer

    # Test 3: Multiple objects
    vol3, lbl3 = create_realistic_synthetic_volume()
    # Add second cube - need to recreate coordinate arrays
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cube_mask2 = ((np.abs(z - 64) < 8) & (np.abs(y - 90) < 8) & (np.abs(x - 90) < 8))
    vol3[cube_mask2] = np.random.normal(0.35, 0.03, np.sum(cube_mask2))
    lbl3[cube_mask2] = 1

    # Test 4: Pure background (no objects)
    vol4 = vol1.copy()
    lbl4 = np.zeros_like(lbl1)
    vol4[lbl1 > 0] = np.random.normal(0.15, 0.05, np.sum(lbl1))  # Replace cube with background

    return [(vol1, lbl1), (vol2, lbl2), (vol3, lbl3), (vol4, lbl4)]


def analyze_model_attention(model, patch, save_path, pos=None):
    """Visualize what the model pays attention to with proper color scales"""
    patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).cuda()
    patch_tensor.requires_grad_(True)

    output = model(patch_tensor)
    loss = output[0, 0].sum()  # Focus on foreground channel
    loss.backward()

    # Gradient-based attention
    gradients = patch_tensor.grad.data.abs()
    attention_map = gradients[0, 0].cpu().numpy()

    # Extract middle slice
    if not pos:
        mid_z = patch.shape[0] // 2
    else:
        mid_z = pos
    input_slice = patch[mid_z]
    true_fg_slice = attention_map[mid_z]
    pred_fg_slice = torch.sigmoid(output)[0, 0, mid_z].detach().cpu().numpy()

    # Create figure with proper colorbars
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original (input intensities)
    im1 = axes[0].imshow(input_slice, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Input Intensity', rotation=270, labelpad=15)

    # Attention map (gradient magnitudes)
    im2 = axes[1].imshow(attention_map[mid_z], cmap='hot')
    axes[1].set_title('Model Attention')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Gradient Magnitude', rotation=270, labelpad=15)

    # Prediction (probabilities 0-1)
    im3 = axes[2].imshow(pred_fg_slice, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Probability', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print value ranges for reference
    print(f"Input range: {input_slice.min():.4f} to {input_slice.max():.4f}")
    print(f"Attention range: {attention_map[mid_z].min():.6f} to {attention_map[mid_z].max():.6f}")
    print(f"Prediction range: {pred_fg_slice.min():.4f} to {pred_fg_slice.max():.4f}")

def test_single_patch(model_path, config_path, volume_path, x,y,z, save):
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
    tifffile.imwrite(save+ f'/single_patch_input{z}_{y}_{x}.tif', patch)
    tifffile.imwrite(save +f'/single_patch_foreground{z}_{y}_{x}.tif', foreground)
    tifffile.imwrite(save + f'/single_patch_boundary{z}_{y}_{x}.tif', boundary)

    print(f"Logit range: {logits.min().item():.2f} to {logits.max().item():.2f}")
    print(f"Probability range: {probs.min().item():.4f} to {probs.max().item():.4f}")
    print(f"Foreground pixels: {np.sum(foreground > 0.5)} / {foreground.size}")

    return patch, foreground, boundary


# Usage
ID  =  37227920
model_path = f"/mnt/duke-netapp/asvetlove/plankton_results/job_{ID}/checkpoints/best_model.pth"
config_path = f"/mnt/duke-netapp/asvetlove/plankton_results/job_{ID}/logs/config.json"
volume_path = "/home/asvetlove/data/segmentation/inference_examples/POR_20to200_20231022_AM_01_epo_02/"
output_dir = f"/mnt/duke-netapp/asvetlove/plankton_results/job_{ID}/tests/"
os.makedirs(output_dir, exist_ok=True)
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
#for _x,_y,_z in zip(x, y, z):
#    test_single_patch(model_path, config_path, volume_path, _x,_y,_z, output_dir)
model = load_trained_model(model_path, config_path)
model.eval()
patch = tifffile.imread('/home/asvetlove/PycharmProjects/TREC_seg_unet/data/ml_patches/POR_20o200_20231022_AM_01_epo_01/patch_0002.tif')
tile_dim = 128
pos_test_1 = [242,380,250]
false_pos_test_1 = [95,0,0]
patch_pos = patch[pos_test_1[0]:(pos_test_1[0]+tile_dim),pos_test_1[1]:(pos_test_1[1]+tile_dim),pos_test_1[2]:(pos_test_1[2]+tile_dim)]
patch_false_pos = patch[false_pos_test_1[0]:(false_pos_test_1[0]+tile_dim),false_pos_test_1[1]:(false_pos_test_1[1]+tile_dim),false_pos_test_1[2]:(false_pos_test_1[2]+tile_dim)]

analyze_model_attention(model, patch_false_pos, output_dir+'training_false_pos_test', pos=72)
analyze_model_attention(model, patch_pos, output_dir+'training_pos_test', pos=2)

# Run comprehensive test
synthetic_tests = test_feature_discrimination()

for i, (vol, lbl) in enumerate(synthetic_tests):
    print(f"Test {i + 1}: Ground truth has {np.sum(lbl)} positive pixels")
#   # Run through your attention analysis
    analyze_model_attention(model, vol, output_dir + f'/synthetic_test_{i + 1}.png')