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
            step_size = min(self.patch_shape) // 4

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

    def __getitem__(self, index):
        """Skip to next volume if current has no valid patches"""
        vol_idx = index // self.samples_per_volume

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
        weights = np.array([loc[3] + 0.1 for loc in locations])
        weights = weights / np.sum(weights)

        chosen_idx = np.random.choice(len(locations), p=weights)
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

        # Convert to tensors and add channel dimension
        image_patch = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
        label_patch = torch.tensor(label_patch, dtype=torch.uint8).unsqueeze(0)

        # Apply transforms if specified
        if self.transform:
            image_patch = tio.ScalarImage(tensor=image_patch)
            label_patch = tio.LabelMap(tensor=label_patch)
            subject = tio.Subject(image=image_patch, label=label_patch)

            transformed = self.transform(subject)
            image_patch = transformed.image.tensor
            label_patch = transformed.label.tensor.squeeze(0)
        else:
            label_patch = label_patch.squeeze(0)

        # Transform mask (create foreground/boundary targets)
        label_patch = self.mask_transform(label_patch)

        return image_patch, label_patch

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