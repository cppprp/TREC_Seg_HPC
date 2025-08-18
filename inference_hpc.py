#!/usr/bin/env python3
"""
HPC-optimized inference script for plankton segmentation
Enhanced version with command-line arguments and improved flexibility
Optimized for A100 GPUs with large memory and fast storage
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json
import gc
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tifffile
import torch
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm

# Add path for imports (same as training)
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from dataset_hpc import fixed_normalise
from model_hpc import load_trained_model


def setup_logging(log_file: Optional[str] = None, verbose: bool = True) -> logging.Logger:
    """Setup logging for HPC environment"""
    logger = logging.getLogger('hpc_inference')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class HPC_InferenceConfig:
    """HPC-optimized inference configuration"""

    def __init__(self, job_id: str, scratch_base: str = "/scratch",
                 home_base: str = "/home", username: str = None):
        # Get username from environment if not provided
        if username is None:
            username = os.environ.get('USER', os.environ.get('USERNAME', 'user'))

        # HPC paths
        self.scratch_base = f"{scratch_base}/{username}"
        self.home_base = f"{home_base}/{username}"
        self.job_id = job_id

        # Model paths
        self.checkpoint_path = f"{self.scratch_base}/plankton_results/job_{self.job_id}/checkpoints/best_model.pth"
        self.config_path = f"{self.scratch_base}/plankton_results/job_{self.job_id}/logs/config.json"

        # HPC-optimized inference parameters
        self.block_shape = (256, 256, 256)  # Will be auto-tuned based on GPU
        self.halo = (64, 64, 64)  # Will be auto-tuned based on GPU
        self.gpu_ids = [0]  # Single GPU (CUDA_VISIBLE_DEVICES handles assignment)

        # I/O optimization
        self.use_scratch_for_temp = True
        self.parallel_io = True
        self.num_io_threads = min(8, os.cpu_count() or 8)
        self.chunk_size = 512

        # Memory optimization
        self.clear_cache_frequency = 3  # Clear GPU cache every N volumes
        self.use_memory_mapping = True
        self.max_memory_gb = None  # Auto-detect

        # Progress tracking
        self.save_progress = True
        self.progress_file = f"{self.scratch_base}/inference_progress_{self.job_id}.json"
        self.checkpoint_frequency = 100  # Save progress every N slices

        # Output configuration
        self.output_format = "tiff"  # or "h5", "zarr"
        self.compression = True

    def auto_tune_gpu_params(self, logger: logging.Logger) -> None:
        """Auto-tune GPU parameters based on available memory"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - using CPU defaults")
            return

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {gpu_memory_gb:.1f} GB")

        # Auto-tune based on GPU memory
        if gpu_memory_gb >= 70:  # A100 80GB
            self.block_shape = (384, 384, 384)
            self.halo = (96, 96, 96)
            self.chunk_size = 1024
        elif gpu_memory_gb >= 35:  # A100 40GB or V100 32GB
            self.block_shape = (256, 256, 256)
            self.halo = (64, 64, 64)
            self.chunk_size = 512
        elif gpu_memory_gb >= 20:  # RTX 3090/4090
            self.block_shape = (192, 192, 192)
            self.halo = (48, 48, 48)
            self.chunk_size = 256
        else:  # Smaller GPUs
            self.block_shape = (128, 128, 128)
            self.halo = (32, 32, 32)
            self.chunk_size = 128

        self.max_memory_gb = gpu_memory_gb * 0.8  # Use 80% of available memory
        logger.info(f"Auto-tuned block shape: {self.block_shape}")
        logger.info(f"Auto-tuned halo: {self.halo}")

    def get_output_paths(self, dataset_name: str) -> Tuple[Path, Path]:
        """Get optimized output paths"""
        base_path = Path(f"{self.scratch_base}/inference_results/job_{self.job_id}/{dataset_name}")

        foreground_path = base_path / "foreground"
        boundary_path = base_path / "boundary"

        return foreground_path, boundary_path

    def validate_paths(self, logger: logging.Logger) -> bool:
        """Validate that required paths exist"""
        errors = []

        if not os.path.exists(self.checkpoint_path):
            errors.append(f"Checkpoint not found: {self.checkpoint_path}")

        if not os.path.exists(self.config_path):
            errors.append(f"Config not found: {self.config_path}")

        # Create necessary directories
        os.makedirs(f"{self.scratch_base}/inference_results/job_{self.job_id}", exist_ok=True)

        if errors:
            for error in errors:
                logger.error(error)
            return False
        return True


class VolumeProcessor:
    """Handles volume loading, processing, and saving"""

    def __init__(self, config: HPC_InferenceConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.temp_dir = None

    def __enter__(self):
        if self.config.use_scratch_for_temp:
            self.temp_dir = tempfile.mkdtemp(dir=f"{self.config.scratch_base}/tmp")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def load_volume(self, volume_path: str, start_slice: Optional[int] = None,
                    num_slices: Optional[int] = None, normalize: bool = False,
                    norm_min: float = -0.01, norm_max: float = 0.025) -> np.ndarray:
        """Load volume with optimizations"""
        self.logger.info(f"Loading volume from: {volume_path}")

        if os.path.isdir(volume_path):
            return self._load_tiff_stack(volume_path, start_slice, num_slices, normalize, norm_min, norm_max)
        else:
            return self._load_single_file(volume_path, normalize, norm_min, norm_max)

    def _load_tiff_stack(self, volume_path: str, start_slice: Optional[int],
                         num_slices: Optional[int], normalize: bool,
                         norm_min: float, norm_max: float) -> np.ndarray:
        """Load TIFF stack with memory optimization"""
        tiff_files = sorted([f for f in os.listdir(volume_path) if f.endswith(('.tif', '.tiff'))])

        if not tiff_files:
            raise ValueError(f"No TIFF files found in {volume_path}")

        # Determine slice range
        if start_slice is not None:
            if num_slices is not None:
                end_slice = min(start_slice + num_slices, len(tiff_files))
            else:
                end_slice = len(tiff_files)
            tiff_files = tiff_files[start_slice:end_slice]
            self.logger.info(f"Processing slices {start_slice} to {end_slice - 1}")

        # Load first image to get dimensions
        first_img = tifffile.imread(os.path.join(volume_path, tiff_files[0]))
        volume_shape = (len(tiff_files), *first_img.shape)

        self.logger.info(f"Volume shape: {volume_shape}")
        self.logger.info(f"Estimated size: {np.prod(volume_shape) * 4 / 1e9:.1f} GB")

        # Use memory mapping for very large volumes
        if self.config.use_memory_mapping and np.prod(volume_shape) > 5e8:  # > 500M voxels
            self.logger.info("Using memory-mapped loading for large volume")
            temp_file = os.path.join(self.temp_dir or '/tmp', f'volume_{os.getpid()}.dat')
            volume = np.memmap(temp_file, dtype=np.float32, mode='w+', shape=volume_shape)
        else:
            volume = np.zeros(volume_shape, dtype=np.float32)

        # Load with progress bar and parallel I/O if beneficial
        for i, filename in enumerate(tqdm(tiff_files, desc="Loading volume")):
            img = tifffile.imread(os.path.join(volume_path, filename))
            volume[i] = img.astype(np.float32)

        # Apply normalization if requested
        if normalize:
            self.logger.info(f"Applying normalization: [{norm_min}, {norm_max}]")
            volume = fixed_normalise(volume, norm_min, norm_max)

        return volume

    def _load_single_file(self, volume_path: str, normalize: bool,
                          norm_min: float, norm_max: float) -> np.ndarray:
        """Load single file volume"""
        self.logger.info("Loading single file volume")
        volume = tifffile.imread(volume_path).astype(np.float32)

        if normalize:
            self.logger.info(f"Applying normalization: [{norm_min}, {norm_max}]")
            volume = fixed_normalise(volume, norm_min, norm_max)

        return volume

    def load_mask(self, mask_path: str, volume_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """Load and prepare mask"""
        if not mask_path or not os.path.exists(mask_path):
            self.logger.warning(f"Mask not found: {mask_path}")
            return None

        self.logger.info(f"Loading mask from: {mask_path}")
        try:
            mask_2d = tifffile.imread(mask_path)

            # Expand to 3D if needed
            if len(volume_shape) == 3 and len(mask_2d.shape) == 2:
                mask = np.broadcast_to(mask_2d[None, :, :], volume_shape)
            else:
                mask = mask_2d

            self.logger.info(f"Mask shape: {mask.shape}")
            return mask

        except Exception as e:
            self.logger.error(f"Failed to load mask: {e}")
            return None

    def save_results(self, foreground: np.ndarray, boundaries: np.ndarray,
                     foreground_path: Path, boundary_path: Path,
                     dataset_name: str) -> None:
        """Save results with parallel I/O"""
        self.logger.info(f"Saving results for {dataset_name}")

        # Create directories
        foreground_path.mkdir(parents=True, exist_ok=True)
        boundary_path.mkdir(parents=True, exist_ok=True)

        if self.config.output_format == "tiff":
            self._save_as_tiff_stack(foreground, boundaries, foreground_path, boundary_path)
        elif self.config.output_format == "h5":
            self._save_as_h5(foreground, boundaries, foreground_path, boundary_path, dataset_name)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")

    def _save_as_tiff_stack(self, foreground: np.ndarray, boundaries: np.ndarray,
                            foreground_path: Path, boundary_path: Path) -> None:
        """Save as TIFF stack with parallel I/O"""

        def save_slice(args):
            slice_idx, fg_slice, bd_slice = args

            # Determine compression
            compress_level = 6 if self.config.compression else 0

            # Save foreground
            fg_file = foreground_path / f"foreground_{slice_idx:04d}.tif"
            tifffile.imwrite(fg_file, fg_slice, photometric='minisblack',
                             compression='zlib' if self.config.compression else None,
                             compressionargs={'level': compress_level})

            # Save boundary
            bd_file = boundary_path / f"boundary_{slice_idx:04d}.tif"
            tifffile.imwrite(bd_file, bd_slice, photometric='minisblack',
                             compression='zlib' if self.config.compression else None,
                             compressionargs={'level': compress_level})

            return slice_idx

        # Prepare data for parallel processing
        save_args = [(i, foreground[i], boundaries[i]) for i in range(foreground.shape[0])]

        # Save in parallel
        num_threads = self.config.num_io_threads if self.config.parallel_io else 1
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(save_slice, save_args),
                      total=len(save_args), desc="Saving slices"))

        self.logger.info(f"Saved {foreground.shape[0]} slices")
        self.logger.info(f"Foreground: {foreground_path}")
        self.logger.info(f"Boundary: {boundary_path}")


def predict_volume_hpc(volume: np.ndarray, model: torch.nn.Module,
                       config: HPC_InferenceConfig, logger: logging.Logger,
                       mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """HPC-optimized prediction with monitoring"""
    logger.info("Running HPC-optimized inference")
    logger.info(f"Block shape: {config.block_shape}")
    logger.info(f"Halo: {config.halo}")

    # Monitor GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"GPU memory before inference: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")

    def sigmoid_postprocess(prediction: torch.Tensor) -> np.ndarray:
        """Apply sigmoid activation and convert to numpy"""
        return torch.sigmoid(prediction).cpu().numpy()

    # Run prediction with monitoring
    try:
        prediction = predict_with_halo(
            volume,
            model,
            gpu_ids=config.gpu_ids,
            block_shape=config.block_shape,
            halo=config.halo,
            mask=mask,
            postprocess=sigmoid_postprocess,
            preprocess=None
        )

        return prediction[0], prediction[1]  # foreground, boundaries

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("GPU out of memory - trying smaller block size")
            # Reduce block size and try again
            original_block = config.block_shape
            config.block_shape = tuple(s // 2 for s in config.block_shape)
            logger.info(f"Retrying with block shape: {config.block_shape}")

            torch.cuda.empty_cache()
            prediction = predict_with_halo(
                volume,
                model,
                gpu_ids=config.gpu_ids,
                block_shape=config.block_shape,
                halo=config.halo,
                mask=mask,
                postprocess=sigmoid_postprocess,
                preprocess=None
            )

            config.block_shape = original_block  # Restore for next volume
            return prediction[0], prediction[1]
        else:
            raise


def process_volume(volume_path: str, mask_path: Optional[str], dataset_name: str,
                   model: torch.nn.Module, config: HPC_InferenceConfig,
                   logger: logging.Logger, **kwargs) -> bool:
    """Process a single volume with full error handling"""

    try:
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        with VolumeProcessor(config, logger) as processor:
            # Load volume
            volume = processor.load_volume(
                volume_path,
                start_slice=kwargs.get('start_slice'),
                num_slices=kwargs.get('num_slices'),
                normalize=kwargs.get('normalize', False),
                norm_min=kwargs.get('norm_min', -0.01),
                norm_max=kwargs.get('norm_max', 0.025)
            )

            load_time = time.time() - start_time
            logger.info(f"Volume loaded in {load_time:.1f}s")

            # Load mask
            mask = processor.load_mask(mask_path, volume.shape) if mask_path else None

            # Run inference
            inference_start = time.time()
            foreground, boundaries = predict_volume_hpc(volume, model, config, logger, mask)
            inference_time = time.time() - inference_start

            logger.info(f"Inference completed in {inference_time:.1f}s")
            logger.info(f"Foreground shape: {foreground.shape}")
            logger.info(f"Boundary shape: {boundaries.shape}")

            # Save results
            save_start = time.time()
            foreground_path, boundary_path = config.get_output_paths(dataset_name)
            processor.save_results(foreground, boundaries, foreground_path, boundary_path, dataset_name)
            save_time = time.time() - save_start

            total_time = time.time() - start_time

            # Performance summary
            logger.info(f"Processing Summary:")
            logger.info(f"  Loading time: {load_time:.1f}s")
            logger.info(f"  Inference time: {inference_time:.1f}s")
            logger.info(f"  Saving time: {save_time:.1f}s")
            logger.info(f"  Total time: {total_time:.1f}s")
            logger.info(f"  Throughput: {volume.shape[0] / total_time:.1f} slices/second")

        # Cleanup
        del volume, foreground, boundaries
        if mask is not None:
            del mask
        gc.collect()
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"Error processing {dataset_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HPC-optimized plankton segmentation inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("job_id",
                        help="Job ID for model checkpoint and config paths")
    parser.add_argument("volume_path",
                        help="Path to volume for prediction (directory for TIFF stack or single file)")

    # Optional arguments
    parser.add_argument("--mask-path",
                        help="Path to mask file (optional)")
    parser.add_argument("--dataset-name",
                        help="Name for output dataset (default: inferred from volume path)")

    # Volume processing options
    parser.add_argument("--start-slice", type=int,
                        help="Starting slice index for partial processing")
    parser.add_argument("--num-slices", type=int,
                        help="Number of slices to process")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply normalization to volume")
    parser.add_argument("--norm-min", type=float, default=-0.01,
                        help="Minimum value for normalization")
    parser.add_argument("--norm-max", type=float, default=0.025,
                        help="Maximum value for normalization")

    # HPC configuration
    parser.add_argument("--scratch-base", default="/scratch",
                        help="Base scratch directory")
    parser.add_argument("--home-base", default="/home",
                        help="Base home directory")
    parser.add_argument("--username",
                        help="Username (default: from environment)")
    parser.add_argument("--num-threads", type=int, metavar="N",
                        help="Number of I/O threads (default: auto-detect from OMP_NUM_THREADS or CPU cores)")
    parser.add_argument("--output-format", choices=["tiff", "h5"], default="tiff",
                        help="Output format")
    parser.add_argument("--no-compression", action="store_true",
                        help="Disable compression")

    # GPU optimization
    parser.add_argument("--block-shape", type=int, nargs=3, metavar=("Z", "Y", "X"),
                        help="Custom block shape (z y x)")
    parser.add_argument("--halo", type=int, nargs=3, metavar=("Z", "Y", "X"),
                        help="Custom halo size (z y x)")

    # Logging and debugging
    parser.add_argument("--log-file",
                        help="Log file path (default: auto-generated)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without running inference")

    return parser.parse_args()


def main():
    """Main function with argument parsing"""
    args = parse_arguments()

    # Setup logging
    if args.log_file is None:
        log_dir = f"{args.scratch_base}/{args.username or os.environ.get('USER', 'user')}/logs"
        os.makedirs(log_dir, exist_ok=True)
        args.log_file = f"{log_dir}/inference_job_{args.job_id}_{int(time.time())}.log"

    logger = setup_logging(args.log_file, args.verbose)

    logger.info("🚀 HPC Plankton Segmentation Inference")
    logger.info("=" * 60)
    logger.info(f"Job ID: {args.job_id}")
    logger.info(f"Volume: {args.volume_path}")
    logger.info(f"Mask: {args.mask_path}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Initialize configuration
        config = HPC_InferenceConfig(args.job_id, args.scratch_base, args.home_base, args.username)

        # Apply command line overrides
        if args.num_threads:
            config.num_io_threads = args.num_threads
        else:
            # Auto-detect optimal number of I/O threads
            # Use OMP_NUM_THREADS if set, otherwise use CPU count
            omp_threads = os.environ.get('OMP_NUM_THREADS')
            if omp_threads:
                config.num_io_threads = min(int(omp_threads), os.cpu_count() or 8)
                logger.info(f"Using OMP_NUM_THREADS for I/O: {config.num_io_threads}")
            else:
                config.num_io_threads = min(8, os.cpu_count() or 8)
                logger.info(f"Auto-detected I/O threads: {config.num_io_threads}")

        config.output_format = args.output_format
        config.compression = not args.no_compression

        # Auto-tune GPU parameters
        config.auto_tune_gpu_params(logger)

        # Apply custom block/halo if specified
        if args.block_shape:
            config.block_shape = tuple(args.block_shape)
            logger.info(f"Using custom block shape: {config.block_shape}")
        if args.halo:
            config.halo = tuple(args.halo)
            logger.info(f"Using custom halo: {config.halo}")

        # Validate configuration
        if not config.validate_paths(logger):
            logger.error("Configuration validation failed")
            return 1

        # Determine dataset name
        dataset_name = args.dataset_name
        if dataset_name is None:
            dataset_name = Path(args.volume_path).stem
            logger.info(f"Auto-detected dataset name: {dataset_name}")

        if args.dry_run:
            logger.info("Dry run - configuration validated successfully")
            return 0

        # Load model
        logger.info("Loading model...")
        logger.info(f"Checkpoint: {config.checkpoint_path}")
        logger.info(f"Config: {config.config_path}")

        model = load_trained_model(config.checkpoint_path, config_path=config.config_path)
        logger.info("✅ Model loaded successfully")

        # Process volume
        start_time = time.time()

        success = process_volume(
            volume_path=args.volume_path,
            mask_path=args.mask_path,
            dataset_name=dataset_name,
            model=model,
            config=config,
            logger=logger,
            start_slice=args.start_slice,
            num_slices=args.num_slices,
            normalize=args.normalize,
            norm_min=args.norm_min,
            norm_max=args.norm_max
        )

        total_time = time.time() - start_time

        if success:
            logger.info("🎉 Inference completed successfully!")
            logger.info(f"⏰ Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
            logger.info(f"📁 Results saved to: {config.scratch_base}/inference_results/job_{args.job_id}/{dataset_name}")
            return 0
        else:
            logger.error("❌ Inference failed")
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())