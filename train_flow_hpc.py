#!/usr/bin/env python3
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import os
import time
import traceback
import atexit


def main():
    """Clean HPC training for plankton segmentation"""

    start_time = time.time()

    try:
        print("=" * 60)
        print("🚀 PLANKTON SEGMENTATION - HPC TRAINING")
        print("=" * 60)
        print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️ Node: {os.getenv('SLURMD_NODENAME', 'local')}")
        print(f"🆔 Job: {os.getenv('SLURM_JOB_ID', 'local')}")

        # Import all required modules
        from config_hpc import setup_config, create_directories, verify_data, get_system_info
        import torch
        from torch.utils.data import DataLoader
        from torch_em.model import UNet3d
        import numpy as np
        import json
        import wandb
        from dataset_hpc import load_and_prepare_data, create_aggressive_transforms, PlanktonDataset
        from model_hpc import WeightedDiceLoss
        from train_epoch_hpc import run_enhanced_training_loop

        print("✅ All modules imported successfully")

        # Setup configuration
        config = setup_config()
        print("✅ Configuration loaded")

        # Create directories
        create_directories(config)

        # Verify data
        verify_data(config)

        # Get system info
        system_info = get_system_info()
        print(f"🎮 GPU: {system_info.get('gpu_name', 'Unknown')}")
        print(f"💾 GPU Memory: {system_info.get('gpu_memory_gb', 0):.1f} GB")

        # Initialize wandb
        try:
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                config=config,
                tags=config['wandb']['tags'],
                notes=config['wandb']['notes'],
                name=f"hpc_job_{config['hpc']['job_id']}"
            )
            wandb.config.update(system_info)
            use_wandb = True
            print("✅ WandB initialized")
        except Exception as e:
            print(f"⚠️ WandB failed: {e}")
            use_wandb = False

        # Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
        else:
            raise RuntimeError("CUDA not available")

        # Save configuration
        config_file = Path(config['paths']['logs']) / 'config.json'
        with open(config_file, 'w') as f:
            # Convert paths to strings for JSON
            config_json = {}
            for key, value in config.items():
                if key == 'paths':
                    config_json[key] = {k: str(v) for k, v in value.items()}
                else:
                    config_json[key] = value
            json.dump(config_json, f, indent=2)
        print("✅ Configuration saved")

        # Load data
        print("📊 Loading training data...")
        train_images, train_labels, val_images, val_labels = load_and_prepare_data(
            config['paths']['training_data'],
            config['training']['train_val_split'],
            config['data']['normalisation_min'],
            config['data']['normalisation_max']
        )

        if len(val_images) == 0:
            print("⚠️ Creating validation split from training data")
            n_val = max(1, len(train_images) // 5)
            val_images = train_images[-n_val:]
            val_labels = train_labels[-n_val:]
            train_images = train_images[:-n_val]
            train_labels = train_labels[:-n_val]

        print(f"📈 Training volumes: {len(train_images)}")
        print(f"📉 Validation volumes: {len(val_images)}")

        # Create datasets
        train_transforms = create_aggressive_transforms()

        train_dataset = PlanktonDataset(
            train_images, train_labels,
            patch_shape=config['training']['patch_shape'],
            transform=train_transforms,
            samples_per_volume=config['training']['samples_per_volume'],
            min_foreground_ratio=config['data']['min_foreground_ratio']
        )

        val_dataset = PlanktonDataset(
            val_images, val_labels,
            patch_shape=config['training']['patch_shape'],
            transform=None,
            samples_per_volume=config['training']['samples_per_volume'] // 2,
            min_foreground_ratio=config['data']['min_foreground_ratio']
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )

        print(f"🔄 Training batches: {len(train_loader)}")
        print(f"🔄 Validation batches: {len(val_loader)}")

        # Create model
        model = UNet3d(**config['model'])
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Model parameters: {total_params:,}")

        # Loss and optimizer
        loss_fn = WeightedDiceLoss(weights=[1.0, 2.0])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, verbose=True
        )

        print("✅ Model, loss, and optimizer ready")

        # Start training
        print("🚀 Starting training...")
        training_start = time.time()

        history = run_enhanced_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=config['training']['n_epochs'],
            device=device,
            save_dir=config['paths']['model'],
            config=config
        )

        training_time = time.time() - training_start
        total_time = time.time() - start_time

        # Results
        best_dice = max(history['val_dice'])
        final_dice = history['val_dice'][-1]

        print("=" * 60)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📊 Best validation Dice: {best_dice:.4f}")
        print(f"📊 Final validation Dice: {final_dice:.4f}")
        print(f"⏰ Training time: {training_time / 3600:.1f} hours")
        print(f"⏰ Total time: {total_time / 3600:.1f} hours")
        print(f"📁 Results: {config['paths']['results_dir']}")

        # Save training summary
        summary = {
            'best_val_dice': best_dice,
            'final_val_dice': final_dice,
            'training_time_hours': training_time / 3600,
            'total_time_hours': total_time / 3600,
            'total_epochs': len(history['train_loss']),
            'job_id': config['hpc']['job_id'],
            'node': config['hpc']['node_name'],
        }

        summary_file = Path(config['paths']['logs']) / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Final wandb log
        if use_wandb:
            wandb.log(summary)
            wandb.finish()

        print("✅ Training completed successfully!")
        print("📁 All results saved to scratch for fast access")

    except Exception as e:
        print("=" * 60)
        print("❌ TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()

        # Save error log
        try:
            error_file = Path(f"/scratch/asvetlove/plankton_results/error_{os.getenv('SLURM_JOB_ID', 'local')}.log")
            error_file.parent.mkdir(parents=True, exist_ok=True)

            with open(error_file, 'w') as f:
                f.write(f"Training Error - Job {os.getenv('SLURM_JOB_ID', 'local')}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)

            print(f"💾 Error saved: {error_file}")
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    # Set seeds
    import torch
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    main()
