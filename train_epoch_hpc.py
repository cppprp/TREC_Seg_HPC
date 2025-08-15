#!/usr/bin/env python3
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from pathlib import Path
import torch
from model_hpc import EarlyStopping, ComprehensiveMetrics
import numpy as np
import json
import wandb
from tqdm import tqdm
from image_log_hpc import ImageLogger, add_image_logging
import time


def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj


def run_enhanced_training_loop(model, train_loader, val_loader, loss_fn,
                               optimizer, scheduler, n_epochs, device, save_dir, config):
    """HPC-optimized training loop with mixed precision and advanced logging"""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize comprehensive tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'learning_rates': [], 'epoch_times': [],
        'gpu_memory_used': [], 'batch_times': []
    }

    # HPC-optimized components
    metrics_calculator = ComprehensiveMetrics()
    early_stopping = EarlyStopping(
        patience=config['optimization']['early_stopping_patience'],
        min_delta=0.001
    )
    best_val_dice = 0.0

    # Mixed precision scaler (HPC always uses AMP)
    scaler = torch.cuda.amp.GradScaler()

    # Enhanced image logger for HPC
    image_logger = ImageLogger(config)

    # Training state tracking
    total_train_time = 0
    total_batches_processed = 0

    print(f"🚀 Starting HPC training loop:")
    print(f"   Mixed Precision: Enabled")
    print(f"   Early Stopping Patience: {config['optimization']['early_stopping_patience']}")
    print(f"   Checkpointing every: {config['training']['checkpoint_every']} epochs")
    print(f"   Learning rate patience: {config['optimization']['lr_scheduler_patience']}")

    for epoch in tqdm(range(n_epochs), desc='HPC Training Progress', ncols=100):
        epoch_start_time = time.time()

        # Training phase with HPC optimizations
        model.train()
        train_losses, train_metrics_list = [], []
        batch_times = []

        train_pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{n_epochs} [Train]',
            leave=False,
            ncols=120
        )

        for batch_idx, (x, y) in enumerate(train_pbar):
            batch_start = time.time()

            optimizer.zero_grad()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Mixed precision forward pass (HPC optimization)
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Gradient clipping before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['gradient_clip']
            )

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            # Record metrics
            train_losses.append(loss.item())
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_batches_processed += 1

            # Calculate training metrics (without gradients)
            with torch.no_grad():
                batch_metrics = metrics_calculator.compute_all_metrics(pred, y)
                train_metrics_list.append(batch_metrics)

            # Update progress bar with HPC stats
            if batch_idx % 5 == 0:
                current_loss = np.mean(train_losses[-5:])
                current_dice = np.mean([m['dice'] for m in train_metrics_list[-5:]])
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                train_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'dice': f'{current_dice:.4f}',
                    'gpu_gb': f'{gpu_mem:.1f}',
                    'batch_ms': f'{batch_time * 1000:.1f}'
                })

        # Validation phase with HPC optimizations
        model.eval()
        val_losses, val_metrics_list = [], []

        val_pbar = tqdm(
            val_loader,
            desc=f'Epoch {epoch + 1}/{n_epochs} [Val]',
            leave=False,
            ncols=120
        )

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_pbar):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                # Mixed precision validation
                with torch.cuda.amp.autocast():
                    pred = model(x)
                    loss = loss_fn(pred, y)

                val_losses.append(loss.item())
                batch_metrics = metrics_calculator.compute_all_metrics(pred, y)
                val_metrics_list.append(batch_metrics)

                # Update validation progress
                if batch_idx % 3 == 0:
                    current_loss = np.mean(val_losses[-3:])
                    current_dice = np.mean([m['dice'] for m in val_metrics_list[-3:]])
                    val_pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'dice': f'{current_dice:.4f}'
                    })

        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        avg_batch_time = np.mean(batch_times)

        # Calculate comprehensive metrics
        train_avg_metrics = {}
        val_avg_metrics = {}

        for metric_name in ['dice', 'iou', 'precision', 'recall', 'f1']:
            train_values = [m[metric_name] for m in train_metrics_list]
            val_values = [m[metric_name] for m in val_metrics_list]

            train_avg_metrics[metric_name] = np.mean(train_values)
            val_avg_metrics[metric_name] = np.mean(val_values)

        # Update comprehensive history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['epoch_times'].append(epoch_time)
        history['batch_times'].append(avg_batch_time)
        history['gpu_memory_used'].append(torch.cuda.memory_allocated() / 1e9)

        for metric_name in ['dice', 'iou', 'precision', 'recall', 'f1']:
            history[f'train_{metric_name}'].append(train_avg_metrics[metric_name])
            history[f'val_{metric_name}'].append(val_avg_metrics[metric_name])

        # Learning rate scheduling
        if scheduler:
            scheduler.step(epoch_val_loss)

        # Model checkpointing with HPC optimizations
        if val_avg_metrics['dice'] > best_val_dice:
            best_val_dice = val_avg_metrics['dice']

            # Save comprehensive checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_dice': best_val_dice,
                'history': history,
                'config': config,
                'hpc_info': {
                    'job_id': config['hpc']['job_id'],
                    'node_name': config['hpc']['node_name'],
                    'total_train_time': total_train_time,
                    'total_batches': total_batches_processed,
                }
            }

            torch.save(checkpoint, save_dir / 'best_model.pth')

            # Save to wandb if enabled
            if config.get('wandb', {}).get('log_model', False):
                try:
                    wandb.save(str(save_dir / 'best_model.pth'))
                except:
                    pass  # Continue if wandb save fails

        # Regular checkpointing for HPC reliability
        if epoch % config['training']['checkpoint_every'] == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'history': history,
                'total_train_time': total_train_time,
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved at epoch {epoch}")

        # Comprehensive progress reporting
        print(f'\n📊 Epoch {epoch + 1}/{n_epochs} Results:')
        print(f'  Time: {epoch_time:.1f}s (avg batch: {avg_batch_time * 1000:.1f}ms)')
        print(f'  Loss - Train: {epoch_train_loss:.4f}, Val: {epoch_val_loss:.4f}')
        print(
            f'  Dice - Train: {train_avg_metrics["dice"]:.4f}, Val: {val_avg_metrics["dice"]:.4f} (Best: {best_val_dice:.4f})')
        print(f'  IoU  - Train: {train_avg_metrics["iou"]:.4f}, Val: {val_avg_metrics["iou"]:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}, GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB')

        # Comprehensive wandb logging
        try:
            log_dict = {
                'epoch': epoch,
                'time/epoch_time': epoch_time,
                'time/avg_batch_time': avg_batch_time,
                'time/total_train_time': total_train_time,
                'train/loss': epoch_train_loss,
                'val/loss': epoch_val_loss,
                'train/dice': train_avg_metrics['dice'],
                'val/dice': val_avg_metrics['dice'],
                'train/iou': train_avg_metrics['iou'],
                'val/iou': val_avg_metrics['iou'],
                'train/precision': train_avg_metrics['precision'],
                'val/precision': val_avg_metrics['precision'],
                'train/recall': train_avg_metrics['recall'],
                'val/recall': val_avg_metrics['recall'],
                'train/f1': train_avg_metrics['f1'],
                'val/f1': val_avg_metrics['f1'],
                'optimization/learning_rate': optimizer.param_groups[0]['lr'],
                'optimization/best_val_dice': best_val_dice,
                'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1e9,
                'system/gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'hpc/batches_processed': total_batches_processed,
                'hpc/avg_batch_time_ms': avg_batch_time * 1000,
            }

            wandb.log(log_dict)
        except Exception as e:
            print(f"⚠️ WandB logging failed: {e}")

        # Enhanced image logging for HPC
        if config.get('logging', {}).get('log_images', True):
            try:
                add_image_logging(
                    image_logger=image_logger,
                    epoch=epoch,
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    config=config
                )
            except Exception as e:
                print(f"⚠️ Image logging failed: {e}")

        # Early stopping check
        if early_stopping(-val_avg_metrics['dice'], model):
            print(f'🛑 Early stopping triggered at epoch {epoch + 1}')
            print(f'   Best validation Dice: {best_val_dice:.4f}')
            break

        # Memory cleanup for long HPC runs
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

    # Training completed - save comprehensive results
    final_history = make_json_serializable(history)

    # Add HPC-specific training summary
    training_summary = {
        'total_epochs': len(history['train_loss']),
        'best_val_dice': best_val_dice,
        'final_val_dice': val_avg_metrics['dice'],
        'total_training_time_hours': total_train_time / 3600,
        'total_batches_processed': total_batches_processed,
        'avg_epoch_time_minutes': np.mean(history['epoch_times']) / 60,
        'avg_batch_time_ms': np.mean(history['batch_times']) * 1000,
        'final_learning_rate': optimizer.param_groups[0]['lr'],
        'hpc_job_id': config['hpc']['job_id'],
        'hpc_node': config['hpc']['node_name'],
    }

    final_history['training_summary'] = training_summary

    with open(save_dir / 'hpc_training_history.json', 'w') as f:
        json.dump(final_history, f, indent=2)

    # Save final model with comprehensive info
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': final_history,
        'config': config,
        'training_summary': training_summary,
        'best_val_dice': best_val_dice,
        'final_val_dice': val_avg_metrics['dice'],
    }
    torch.save(final_checkpoint, save_dir / 'final_model.pth')

    print(f"\n✅ HPC training loop completed successfully!")
    print(f"📊 Training Summary:")
    for key, value in training_summary.items():
        if key not in ['hpc_job_id', 'hpc_node']:
            print(f"   {key}: {value}")

    return final_history