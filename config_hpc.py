#!/usr/bin/env python3
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import os
from pathlib import Path


def setup_config():
    """Clean HPC configuration - everything on scratch"""

    # Get job info
    job_id = os.getenv('SLURM_JOB_ID', 'local')
    node_name = os.getenv('SLURMD_NODENAME', 'local')

    # Everything on scratch for speed
    scratch_base = "/scratch/asvetlove"
    training_data = f"{scratch_base}/ML_training_data"
    results_dir = f"{scratch_base}/plankton_results/job_{job_id}"

    config = {
        'model': {
            'in_channels': 1,
            'out_channels': 2,
            'initial_features': 32,
            'final_activation': None
        },
        'training': {
            'batch_size': 6,  # Optimized for A100
            'learning_rate': 3e-5,
            'n_epochs': 200,
            'patch_shape': (128, 128, 128),
            'samples_per_volume': 20,
            'train_val_split': 0.8,
            'checkpoint_every': 20,
            'loss': 'focal',
            'gradient_clip': 1.0,
            'num_workers': 12,
            'pin_memory': True,
            'mixed_precision': True,
        },
        'loss_specs': {
            'wdice_weights': [1.0, 2.0],
            'wdice_smooth': 1e-7,
            'focal_alpha': 0.22733153821514615,
            'focal_gamma': 3,
            'hard_neg_focal_ratio': 2.0,
            'tversky_focal_alpha': 0.3,
            'tversky_focal_beta': 0.7,
            'tversky_focal_gamma': 2.0,
            'tversky_focal_smooth': 1e-7,
        },
        'optimization': {
            'early_stopping_patience': 20,
            'min_delta': 0.001,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'mode': 'min',
            'factor': 0.5,
            'patience': 15
        },
        'data': {
            'voxel_size_nm': 650,
            'plankton_size_um': (20, 200),
            'min_foreground_ratio': 0.2,
            'normalisation_min': -0.01,
            'normalisation_max': 0.025,
        },
        'paths': {
            'training_data': training_data,
            'results_dir': results_dir,
            'model': f"{results_dir}/checkpoints",
            'logs': f"{results_dir}/logs",
        },
        'wandb': {
            'project': 'plankton-segmentation-hpc',
            'entity': None,
            'tags': ['unet3d', 'hpc', f'job-{job_id}'],
            'notes': f'HPC run - Job {job_id} on {node_name}',
            'log_model': True,
        },
        'logging': {
            'log_images': True,
            'val_log_frequency': 10,
            'progress_log_frequency': 10,
            'num_samples_per_log': 2,
            'log_dataset_overview': True,
        },
        'hpc': {
            'job_id': job_id,
            'node_name': node_name,
        }
    }

    return config


def create_directories(config):
    """Create necessary directories"""
    for key in ['results_dir', 'model', 'logs']:
        path = Path(config['paths'][key])
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {path}")


def verify_data(config):
    """Verify training data exists"""
    data_path = Path(config['paths']['training_data'])

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Count data files
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    tiff_files = list(data_path.rglob("*.tif*"))

    print(f"📊 Data verification:")
    print(f"   Path: {data_path}")
    print(f"   Subdirectories: {len(subdirs)}")
    print(f"   TIFF files: {len(tiff_files)}")

    if len(tiff_files) == 0:
        raise ValueError("No TIFF files found in training data directory")

    return True


def get_system_info():
    """Get system information for logging"""
    import torch

    info = {
        'job_id': os.getenv('SLURM_JOB_ID', 'local'),
        'node': os.getenv('SLURMD_NODENAME', 'local'),
        'user': os.getenv('USER', 'unknown'),
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
        })

    return info
