#!/usr/bin/env python3
import sys
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk, ball
import torchio as tio
import os
import tifffile as tiff
import warnings
# supress torchio warning because annoying
warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader")

class PlanktonDataset(Dataset):
    def __init__(self, images, labels, patch_shape, mask_transform=None,
                 transform=None, samples_per_volume=10, min_foreground_ratio=0.01):
        """
        Improved dataset class for plankton segmentation

        Args:
            images: List of 3D image volumes
            labels: List of 3D label volumes
            patch_shape: Tuple of (D, H, W) for patch dimensions
            mask_transform: Function to transform masks (e.g., create foreground/boundary)
            transform: Augmentation transforms (should be None for validation)
            samples_per_volume: Number of patches to sample per volume per epoch
            min_foreground_ratio: Minimum ratio of foreground pixels in patch
        """
        self.images = images
        self.labels = labels
        self.patch_shape = patch_shape
        self.transform = transform
        self.mask_transform = mask_transform or self.default_mask_transform
        self.samples_per_volume = samples_per_volume
        self.min_foreground_ratio = min_foreground_ratio
        #self.normalisation_min = norm_min
        #self.normalisation_max = norm_max
        self.volume_stats = self._analyze_volume_distributions()
        self.sampling_stats = {
            'fg_ratios': [],
            'volume_usage': np.zeros(len(images)),
            'category_counts': {'high': 0, 'medium': 0, 'low': 0, 'background': 0},
            'total_samples': 0
        }
        # Pre-calculate valid patch locations for each volume
        self.valid_locations = self._find_valid_patch_locations()

    def __len__(self):
        return len(self.images) * self.samples_per_volume

    def _find_valid_patch_locations(self):
        """Find valid patch locations - MINIMAL FIX VERSION"""
        valid_locations = []

        for vol_idx, (image, label) in enumerate(zip(self.images, self.labels)):
            vol_locations = []

            # Calculate possible patch positions
            max_z = image.shape[0] - self.patch_shape[0]
            max_y = image.shape[1] - self.patch_shape[1]
            max_x = image.shape[2] - self.patch_shape[2]

            if max_z <= 0 or max_y <= 0 or max_x <= 0:
                print(f"Warning: Volume {vol_idx} too small for patch size")
                # ADD EMPTY LIST instead of skipping
                valid_locations.append([])
                continue

            # Sample grid of positions and check foreground content
            step_size = min(self.patch_shape) // 2

            for z in range(0, max_z, step_size):
                for y in range(0, max_y, step_size):
                    for x in range(0, max_x, step_size):
                        patch_label = label[z:z + self.patch_shape[0],
                                      y:y + self.patch_shape[1],
                                      x:x + self.patch_shape[2]]

                        fg_ratio = np.sum(patch_label > 0) / patch_label.size
                        if fg_ratio >= self.min_foreground_ratio:
                            vol_locations.append((z, y, x, fg_ratio))

            # ALWAYS add the list (even if empty) to maintain indexing
            valid_locations.append(vol_locations)

            if vol_locations:
                print(f"Volume {vol_idx}: Found {len(vol_locations)} valid patches")
            else:
                print(f"Warning: Volume {vol_idx}: No valid patches found")

        return valid_locations

    def _analyze_volume_distributions(self):
        """Quick sampling to understand what's available in each volume"""
        stats = []

        for vol_idx, (image, label) in enumerate(zip(self.images, self.labels)):
            if any(image.shape[i] < self.patch_shape[i] for i in range(3)):
                stats.append({'fg_ratios': []})
                continue

            fg_ratios = []
            # Sample 50 random patches to understand this volume's distribution
            for _ in range(50):
                z = np.random.randint(0, image.shape[0] - self.patch_shape[0])
                y = np.random.randint(0, image.shape[1] - self.patch_shape[1])
                x = np.random.randint(0, image.shape[2] - self.patch_shape[2])

                patch_label = label[z:z + self.patch_shape[0],
                              y:y + self.patch_shape[1],
                              x:x + self.patch_shape[2]]
                fg_ratio = np.sum(patch_label > 0) / patch_label.size
                fg_ratios.append(fg_ratio)

            stats.append({
                'fg_ratios': fg_ratios,
                'has_background': np.sum(np.array(fg_ratios) < 0.01) > 5,
                'has_medium': np.sum((np.array(fg_ratios) >= 0.05) & (np.array(fg_ratios) < 0.2)) > 5,
                'has_high': np.sum(np.array(fg_ratios) >= 0.2) > 5,
            })

        return stats
    def __getitem__(self, index):
        """Skip to next volume if current has no valid patches"""
        vol_idx = index // self.samples_per_volume
        vol_stats = self.volume_stats[vol_idx]
        # Find a volume with valid locations
        attempts = 0
        max_attempts = len(self.valid_locations)

        while attempts < max_attempts:
            locations = self.valid_locations[vol_idx]

            # If this volume has valid patches, use it
            if locations:
                break

            # Otherwise try next volume
            vol_idx = (vol_idx + 1) % len(self.valid_locations)
            attempts += 1

        # If no volumes have valid patches (shouldn't happen), raise error
        if not locations:
            raise RuntimeError("No volumes with valid patches found!")

        # Weighted sampling based on foreground content
        #weights = np.array([loc[3] + 0.1 for loc in locations])
        #weights = weights / np.sum(weights)

        chosen_idx = np.random.choice(len(locations)) #p=weights)
        z, y, x, _ = locations[chosen_idx]

        # Extract patches
        image = self.images[vol_idx]
        label = self.labels[vol_idx]

        image_patch = image[z:z + self.patch_shape[0],
                      y:y + self.patch_shape[1],
                      x:x + self.patch_shape[2]]
        label_patch = label[z:z + self.patch_shape[0],
                      y:y + self.patch_shape[1],
                      x:x + self.patch_shape[2]]
        '''if not vol_stats['fg_ratios']:  # Empty volume
            return self.__getitem__((index + 1) % len(self))

            # Just sample completely randomly - the class imbalance handling
            # happens at the loss function level, not sampling level
        image = self.images[vol_idx]
        label = self.labels[vol_idx]

        z = np.random.randint(0, image.shape[0] - self.patch_shape[0])
        y = np.random.randint(0, image.shape[1] - self.patch_shape[1])
        x = np.random.randint(0, image.shape[2] - self.patch_shape[2])

        image_patch = image[z:z + self.patch_shape[0],
                      y:y + self.patch_shape[1],
                      x:x + self.patch_shape[2]]
        label_patch = label[z:z + self.patch_shape[0],
                      y:y + self.patch_shape[1],
                      x:x + self.patch_shape[2]]

        fg_ratio = np.sum(label_patch > 0) / label_patch.size
'''
        # Convert to tensors and add channel dimension
        image_patch = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
        label_patch = torch.tensor(label_patch, dtype=torch.uint8).unsqueeze(0)



        '''# Update stats
        self.sampling_stats['fg_ratios'].append(fg_ratio)
        self.sampling_stats['volume_usage'][vol_idx] += 1
        self.sampling_stats['total_samples'] += 1

        # Categorize sample
        if fg_ratio >= 0.2:
            category = 'high'
        elif fg_ratio >= 0.05:
            category = 'medium'
        elif fg_ratio >= 0.01:
            category = 'low'
        else:
            category = 'background'

        self.sampling_stats['category_counts'][category] += 1

        # Print periodic updates
        if self.sampling_stats['total_samples'] % 100 == 0:
            self._print_stats()
        # Apply transforms if specified
        if self.transform:
            image_patch = tio.ScalarImage(tensor=image_patch)
            label_patch = tio.LabelMap(tensor=label_patch)
            subject = tio.Subject(image=image_patch, label=label_patch)

            transformed = self.transform(subject)
            image_patch = transformed.image.tensor
            label_patch = transformed.label.tensor.squeeze(0)
        else:
            label_patch = label_patch.squeeze(0)'''


        # Transform mask (create foreground/boundary targets)
        label_patch = self.mask_transform(label_patch)

        return image_patch, label_patch

    def _print_stats(self):
        """Print sampling statistics"""
        stats = self.sampling_stats
        total = stats['total_samples']

        if total == 0:
            return

        print(f"\n--- Sampling Stats (n={total}) ---")

        # Category distribution
        for cat, count in stats['category_counts'].items():
            pct = count / total * 100
            print(f"{cat}: {count} ({pct:.1f}%)")

        # Foreground ratio stats
        fg_ratios = np.array(stats['fg_ratios'])
        print(f"FG ratio: mean={fg_ratios.mean():.3f}, "
              f"median={np.median(fg_ratios):.3f}, "
              f"std={fg_ratios.std():.3f}")

        # Volume usage
        vol_usage = stats['volume_usage']
        active_vols = np.sum(vol_usage > 0)
        print(f"Active volumes: {active_vols}/{len(vol_usage)}")
        print(f"Volume usage: min={vol_usage.min():.0f}, "
              f"max={vol_usage.max():.0f}, "
              f"mean={vol_usage.mean():.1f}")

        # Identify unused volumes
        unused = np.where(vol_usage == 0)[0]
        if len(unused) > 0 and len(unused) < 10:
            print(f"Unused volumes: {unused.tolist()}")
        elif len(unused) >= 10:
            print(f"Unused volumes: {len(unused)} total")
    @staticmethod
    def default_mask_transform(mask):
        """Create foreground and boundary targets"""
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

        # Foreground (any labeled region)
        foreground = (mask_np > 0).astype(np.float32)

        # Boundaries - use thicker boundaries for better training
        boundaries = np.zeros_like(foreground, dtype=np.float32)

        # Find boundaries for each unique label
        unique_labels = np.unique(mask_np)
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue

            label_mask = (mask_np == label_id)
            label_boundaries = find_boundaries(label_mask, mode='thick')
            # Dilate boundaries slightly for better learning
            label_boundaries = binary_dilation(label_boundaries, ball(1))
            boundaries = np.logical_or(boundaries, label_boundaries)

        boundaries = boundaries.astype(np.float32)

        return torch.stack([torch.tensor(foreground), torch.tensor(boundaries)])

class TiledValidationDataset(Dataset):
    """Validation dataset that uses systematic tiling like inference"""

    def __init__(self, images, labels, tile_shape=(128, 128, 128), halo=32,
                 mask_transform=None, min_foreground=100, tiles = 30):
        self.images = images
        self.labels = labels
        self.tile_shape = tile_shape
        self.halo = halo
        self.mask_transform = mask_transform or PlanktonDataset.default_mask_transform
        self.min_foreground = min_foreground
        self.max_tiles = tiles

        # Pre-compute all tile positions
        self.tile_positions = self._compute_tile_positions()
        print(f"Created {len(self.tile_positions)} validation tiles")

    def _compute_tile_positions(self):
        """Compute systematic tile positions across all volumes"""
        positions = []

        for vol_idx, (image, label) in enumerate(zip(self.images, self.labels)):
            vol_tiles = 0

            # Systematic tiling (same as inference)
            for z in range(0, image.shape[0] - self.tile_shape[0] + 1, self.tile_shape[0]):
                for y in range(0, image.shape[1] - self.tile_shape[1] + 1, self.tile_shape[1]):
                    for x in range(0, image.shape[2] - self.tile_shape[2] + 1, self.tile_shape[2]):

                        # Check if tile has enough foreground
                        tile_label = label[z:z + self.tile_shape[0],
                                     y:y + self.tile_shape[1],
                                     x:x + self.tile_shape[2]]

                        if np.sum(tile_label > 0) >= self.min_foreground and len(positions)< self.max_tiles:
                            positions.append((vol_idx, z, y, x))
                            vol_tiles += 1

            print(f"Volume {vol_idx}: {vol_tiles} validation tiles")

        return positions

    def __len__(self):
        return len(self.tile_positions)

    def __getitem__(self, index):
        vol_idx, z, y, x = self.tile_positions[index]

        image = self.images[vol_idx]
        label = self.labels[vol_idx]

        # Extract tile with halo (same as inference)
        z_start = max(0, z - self.halo)
        y_start = max(0, y - self.halo)
        x_start = max(0, x - self.halo)
        z_end = min(image.shape[0], z + self.tile_shape[0] + self.halo)
        y_end = min(image.shape[1], y + self.tile_shape[1] + self.halo)
        x_end = min(image.shape[2], x + self.tile_shape[2] + self.halo)

        image_tile = image[z_start:z_end, y_start:y_end, x_start:x_end]
        label_tile = label[z_start:z_end, y_start:y_end, x_start:x_end]

        # Pad to consistent size
        target_shape = (self.tile_shape[0] + 2 * self.halo,
                        self.tile_shape[1] + 2 * self.halo,
                        self.tile_shape[2] + 2 * self.halo)

        def pad_to_shape(arr, target_shape):
            pad_width = []
            for i in range(len(target_shape)):
                diff = target_shape[i] - arr.shape[i]
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            return np.pad(arr, pad_width, mode='reflect' if arr.dtype != np.uint8 else 'constant')

        if image_tile.shape != target_shape:
            image_tile = pad_to_shape(image_tile, target_shape)
            label_tile = pad_to_shape(label_tile, target_shape)

        # Convert to tensors
        image_tensor = torch.tensor(image_tile, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_tile, dtype=torch.uint8).unsqueeze(0)

        # Transform labels
        label_tensor = self.mask_transform(label_tensor.squeeze(0))

        # Create center mask for loss computation
        center_mask = torch.zeros_like(label_tensor[0], dtype=torch.bool)
        center_mask[self.halo:-self.halo, self.halo:-self.halo, self.halo:-self.halo] = True

        return image_tensor, label_tensor
def create_transforms():
    """Create augmentation transforms for training"""

    train_transforms = tio.Compose([
        tio.RandomAffine(
            scales=(0.8, 1.2),
            degrees=15,
            translation=5,
            p=0.7
        ),
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomNoise(std=0.1, p=0.3),
        tio.RandomBlur(std=(0, 1), p=0.2),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
    ])

    return train_transforms

def create_aggressive_transforms():
    """More aggressive augmentation for small datasets"""
    return tio.Compose([
        tio.RandomAffine(scales=(0.7, 1.3), degrees=20, translation=10, p=0.8),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5, p=0.3),
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomNoise(std=(0, 0.15), p=0.4),
        tio.RandomBlur(std=(0, 1.5), p=0.3),
        tio.RandomGamma(log_gamma=(-0.4, 0.4), p=0.4),
        tio.RandomBiasField(coefficients=0.5, p=0.2),
    ])

def normalise(image):
    image = image.astype("float32")
    # Normalise the image to roughly [0,1]
    minim = np.min(image)
    image = image - minim
    # We don't use the max value here, because there are a few very bright
    # pixels in some images, that would otherwise throw off the normalization.
    # Instead we use the 95th percentile to be robust against these intensity outliers.
    max_value = np.percentile(image, 95)
    image /= max_value
    image[np.isnan(image)] = 0.0
    return image


def fixed_normalise(image, _min, _max):
    """Fixed-range normalization for consistent training"""
    image = image.astype("float32")

    # Use your tissue intensity range (not per-image percentiles)
    #_min = -0.005
    #_max = 0.010

    image = (image - _min) / (_max - _min)
    # Clip to meaningful range
    image = np.clip(image, 0, 1)

    # Handle any remaining NaN values
    image[np.isnan(image)] = 0.0

    return image

def load_and_prepare_data(data_dir, split, _min, _max):
    """Load and prepare training data with proper volume-level splitting"""

    print("Loading data...")
    images, labels = [], []

    # Load all volumes
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    #subfolders = subfolders[4:6] #<-----------------------------------------------------Here I am
    for i, folder in enumerate(subfolders):
        print(f"Loading volume {i + 1}/{len(subfolders)}: {folder}")

        label_files = [f.path for f in os.scandir(folder) if 'label' in f.path]

        for label_file in label_files:
            image_file = label_file.replace('.labels', '')

            if os.path.exists(image_file):
                # Load and normalize image
                image = fixed_normalise(tiff.imread(image_file), _min, _max)
                label = tiff.imread(label_file)

                # Basic quality checks
                if image.shape != label.shape:
                    print(f"Warning: Shape mismatch in {image_file}")
                    continue

                if np.sum(label > 0) < 100:  # Very sparse labels
                    print(f"Warning: Very sparse labels in {label_file}")
                    continue

                images.append(image)
                labels.append(label)

    print(f"Loaded {len(images)} volumes")

    # Volume-level train/val split (prevents data leakage)
    n_train = int(len(images) * split)

    # Shuffle indices for random split
    indices = np.random.permutation(len(images))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"Training volumes: {len(train_images)}")
    print(f"Validation volumes: {len(val_images)}")

    return train_images, train_labels, val_images, val_labels


def background_aware_normalize(volume):
    """Normalize while preserving background structure"""

    print("🔧 Applying background-aware normalization...")
    volume = volume.astype(np.float32)

    # Identify background (near-zero values)
    #background_threshold = 0.000
    background_mask = np.abs(volume) == 0.000
    foreground_mask = ~background_mask

    print(f"   Background pixels: {np.sum(background_mask):,} ({np.sum(background_mask) / volume.size * 100:.1f}%)")
    print(f"   Foreground pixels: {np.sum(foreground_mask):,}")

    # Normalize only foreground pixels
    fg_values = volume[foreground_mask]
    fg_min, fg_max = np.percentile(fg_values, [1, 99])  # More robust percentiles

    print(f"   Foreground range: {fg_min:.6f} to {fg_max:.6f}")

    normalized = np.zeros_like(volume)

    if fg_max > fg_min:
        # Map foreground to 0.1-0.9 (leaving 0-0.1 for background)
        fg_normalized = 0.1 + 0.8 * (np.clip(fg_values, fg_min, fg_max) - fg_min) / (fg_max - fg_min)
        normalized[foreground_mask] = fg_normalized

    # Background stays near 0
    normalized[background_mask] = 0.00  # Small positive value

    print(f"   Final range: {normalized.min():.6f} to {normalized.max():.6f}")
    print(f"   Background normalized to: {normalized[background_mask].mean():.6f}")
    print(
        f"   Foreground normalized to: {normalized[foreground_mask].min():.6f} - {normalized[foreground_mask].max():.6f}")

    return normalized


def report_epoch_sampling_stats(train_dataset, epoch):
    """Report what the model trained on this epoch"""
    stats = train_dataset.sampling_stats
    total = stats['total_samples']

    if total == 0:
        return

    print(f"\n=== EPOCH {epoch} SAMPLING REPORT ===")

    # Reset for next epoch tracking
    epoch_samples = total - getattr(train_dataset, '_last_epoch_total', 0)
    train_dataset._last_epoch_total = total

    print(f"Samples this epoch: {epoch_samples}")

    # Category breakdown
    for cat, count in stats['category_counts'].items():
        pct = count / total * 100
        print(f"  {cat}: {pct:.1f}%")

    # Check for problems
    bg_pct = stats['category_counts']['background'] / total * 100
    if bg_pct < 5:
        print("  ⚠️  WARNING: Very few background samples (<5%)")

    unused_volumes = np.sum(stats['volume_usage'] == 0)
    if unused_volumes > len(stats['volume_usage']) * 0.3:
        print(f"  ⚠️  WARNING: {unused_volumes} volumes unused (>{30}%)")

    print("=" * 40)