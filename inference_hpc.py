#!/usr/bin/env python3
"""
Optimized simple inference script
- Always normalizes data
- Optimized I/O for reading and writing
- Configurable block and halo sizes
"""

import os
import sys
import argparse
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm

# Add path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from dataset_hpc import fixed_normalise
from model_hpc import load_trained_model


def load_tif_stack_optimized(volume_path, start_slice=None, num_slices=None, bit=32):
    """
    Optimized TIFF stack loading with memory management
    """
    print(f"📊 Loading volume from: {volume_path}")

    if os.path.isfile(volume_path):
        # Single TIFF file
        print("   Loading single TIFF file")
        volume = tifffile.imread(volume_path)
        return volume.astype(np.float32)

    # Directory of TIFF files
    tiff_files = sorted([f for f in os.listdir(volume_path)
                         if f.lower().endswith(('.tif', '.tiff'))])

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {volume_path}")

    print(f"   Found {len(tiff_files)} TIFF files")

    # Determine slice range
    if start_slice is not None:
        if num_slices is not None:
            end_slice = min(start_slice + num_slices, len(tiff_files))
        else:
            end_slice = len(tiff_files)
        tiff_files = tiff_files[start_slice:end_slice]
        print(f"   Processing slices {start_slice} to {end_slice - 1}")

    # Load first image to get dimensions
    first_img = tifffile.imread(os.path.join(volume_path, tiff_files[0]))
    volume_shape = (len(tiff_files), *first_img.shape)

    print(f"   Volume shape: {volume_shape}")
    print(f"   Estimated size: {np.prod(volume_shape) * 4 / 1e9:.1f} GB")

    # Allocate volume
    volume = np.zeros(volume_shape, dtype=np.float32)

    # Load with progress bar
    for i, filename in enumerate(tqdm(tiff_files, desc="Loading slices")):
        img = tifffile.imread(os.path.join(volume_path, filename))
        volume[i] = img.astype(np.float32)

        # Periodic garbage collection for large volumes
        if i % 100 == 0:
            gc.collect()

    return volume

def load_tif_stack(input_folder, start=None, chunk=None, bit=16):
    """Load TIFF stack with memory monitoring"""
    data = []
    exclusion_criteria = ['pre', '._', '.DS', 'overlaps']

    file_names = os.listdir(input_folder)
    fnames = sorted([file for file in file_names if
                     not any(exclude_str in file for exclude_str in exclusion_criteria) and file.endswith('.tif')])

    if start is None:
        start = 0
    if chunk is None:
        chunk = len(fnames)

    fnames = fnames[start:start + chunk]
    print(f"Loading {len(fnames)} TIFF files...")

    for fname in tqdm(fnames, desc=f"Loading data from {input_folder}"):
        image_path = os.path.join(input_folder, fname)
        data.append(tifffile.imread(image_path))

    if bit == 32:
        data = np.asarray(data, dtype='float32')
    elif bit == 16:
        data = np.asarray(data, dtype='uint16')
    elif bit == 8:
        data = np.asarray(data, dtype='uint8')

    print(f"Loaded volume shape: {data.shape}")
    print(f"Volume memory usage: {data.nbytes / 1e9:.2f} GB")

    return data
def save_tiff_stack_parallel(data, output_path, prefix, num_threads=8):
    """
    Optimized parallel TIFF stack writing
    """
    print(f"💾 Saving {prefix} to {output_path} (using {num_threads} threads)")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    def save_slice(slice_idx):
        filename = f"{prefix}_{slice_idx:04d}.tif"
        filepath = os.path.join(output_path, filename)
        tifffile.imwrite(
            filepath,
            data[slice_idx],
            photometric='minisblack',
            compression='zlib',
            compressionargs={'level': 6}
        )
        return slice_idx

    # Save slices in parallel
    slice_indices = range(data.shape[0])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(
            executor.map(save_slice, slice_indices),
            total=len(slice_indices),
            desc=f"Saving {prefix}"
        ))

    print(f"   ✅ Saved {data.shape[0]} slices")


def sigmoid_postprocess(prediction):
    """Apply sigmoid activation and convert to numpy"""
    import torch
    return torch.sigmoid(prediction).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Optimized plankton inference with configurable parameters"
    )

    # Required arguments
    parser.add_argument("job_id", help="Job ID for model paths")
    parser.add_argument("volume_path", help="Path to volume (directory or single file)")

    # Optional arguments
    parser.add_argument("--mask-path", help="Path to mask file")
    parser.add_argument("--dataset-name", help="Dataset name for output")
    parser.add_argument("--start-slice", type=int, help="Starting slice index")
    parser.add_argument("--num-slices", type=int, help="Number of slices to process")

    # Normalization (always applied)
    parser.add_argument("--norm-min", type=float, default=-0.01,
                        help="Normalization minimum value")
    parser.add_argument("--norm-max", type=float, default=0.025,
                        help="Normalization maximum value")

    # Prediction parameters
    parser.add_argument("--block-shape", type=int, nargs=3, default=[128, 128, 128],
                        metavar=("Z", "Y", "X"), help="Block shape for prediction")
    parser.add_argument("--halo", type=int, nargs=3, default=[32, 32, 32],
                        metavar=("Z", "Y", "X"), help="Halo size for prediction")

    # I/O optimization
    parser.add_argument("--num-threads", type=int, default=8,
                        help="Number of threads for parallel I/O")

    args = parser.parse_args()

    # Convert lists to tuples
    block_shape = tuple(args.block_shape)
    halo = tuple(args.halo)

    print("🚀 Optimized Plankton Inference")
    print("=" * 50)
    print(f"Job ID: {args.job_id}")
    print(f"Volume: {args.volume_path}")
    print(f"Block shape: {block_shape}")
    print(f"Halo: {halo}")
    print(f"Normalization: [{args.norm_min}, {args.norm_max}]")
    print("=" * 50)

    # Build model paths
    checkpoint_path = f"/scratch/asvetlove/plankton_results/job_{args.job_id}/checkpoints/best_model.pth"
    config_path = f"/scratch/asvetlove/plankton_results/job_{args.job_id}/logs/config.json"

    print(f"📥 Loading model...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Config: {config_path}")

    # Verify files exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load model
    start_time = time.time()
    model = load_trained_model(checkpoint_path, config_path=config_path)
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s")

    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = Path(args.volume_path).stem

    print(f"\n📊 Processing dataset: {dataset_name}")

    # Load volume
    start_time = time.time()
    volume = load_tif_stack(
        args.volume_path,
        start=args.start_slice,
        chunk=args.num_slices, bit = 32
    )
    load_time = time.time() - start_time
    print(f"📊 Volume loaded in {load_time:.1f}s")

    # Apply normalization (always)
    print(f"🔧 Applying normalization: [{args.norm_min}, {args.norm_max}]")
    volume = fixed_normalise(volume, args.norm_min, args.norm_max)

    # Load mask if provided
    mask = None
    if args.mask_path and os.path.exists(args.mask_path):
        print(f"🎭 Loading mask: {args.mask_path}")
        mask_2d = tifffile.imread(args.mask_path)
        mask = np.array([mask_2d] * volume.shape[0])
        print(f"   Mask shape: {mask.shape}")
    else:
        print("   No mask provided")

    # Run prediction
    print(f"\n🧠 Running prediction...")
    print(f"   Block shape: {block_shape}")
    print(f"   Halo: {halo}")

    start_time = time.time()
    prediction = predict_with_halo(
        volume,
        model,
        gpu_ids=[0],
        block_shape=block_shape,
        halo=halo,
        mask=mask,
        postprocess=sigmoid_postprocess,
        preprocess=None
    )
    inference_time = time.time() - start_time

    foreground, boundaries = prediction[0], prediction[1]
    print(f"🧠 Prediction completed in {inference_time:.1f}s")
    print(f"   Foreground shape: {foreground.shape}")
    print(f"   Boundary shape: {boundaries.shape}")

    # Setup output paths
    output_base = f"/scratch/asvetlove/inference_results/job_{args.job_id}"
    foreground_path = f"{output_base}/{dataset_name}/foreground/"
    boundary_path = f"{output_base}/{dataset_name}/boundary/"

    print(f"\n💾 Saving results:")
    print(f"   Foreground: {foreground_path}")
    print(f"   Boundary: {boundary_path}")

    # Save results in parallel
    start_time = time.time()
    save_tiff_stack_parallel(foreground, foreground_path, "foreground", args.num_threads)
    save_tiff_stack_parallel(boundaries, boundary_path, "boundary", args.num_threads)
    save_time = time.time() - start_time

    # Performance summary
    total_time = load_time + inference_time + save_time
    print(f"\n📈 Performance Summary:")
    print(f"   Loading time: {load_time:.1f}s")
    print(f"   Inference time: {inference_time:.1f}s")
    print(f"   Saving time: {save_time:.1f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Throughput: {volume.shape[0] / total_time:.1f} slices/second")

    # Cleanup
    del volume, foreground, boundaries, prediction
    if mask is not None:
        del mask
    gc.collect()

    print("✅ Inference complete!")


if __name__ == "__main__":
    main()
