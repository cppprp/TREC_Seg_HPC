#!/usr/bin/env python3
import sys
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
import wandb
import torch
import numpy as np
import time


class ImageLogger:
    def __init__(self, config):
        self.config = config
        self.log_config = config.get('logging', {})

        # Data storage
        self.progress_data = []  # Same samples over time
        self.validation_data = []  # Different samples each epoch
        self.tracking_samples = []  # Fixed samples for progress tracking
        self.initialized = False

        self.progress_table = wandb.Table(
            columns=["epoch", "sample_id", "category", "dice", "iou", "fg_ratio", "overlay"],
            log_mode='INCREMENTAL'
        )
        self.validation_table = wandb.Table(
            columns=["epoch", "sample_id", "category", "dice", "iou", "fg_ratio", "overlay"],
            log_mode='INCREMENTAL'
        )

    def normalize_image(self, image):
        """Simple image normalization"""
        img = image.astype(np.float32)
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip(img, p2, p98)
        if p98 > p2:
            img = (img - p2) / (p98 - p2) * 255
        return img.astype(np.uint8)

    def create_colored_overlay(self, background, pred_fg, pred_bd, true_fg, true_bd, title=""):
        """Create overlay with all channels - wandb auto-assigns distinct colors"""
        bg_normalized = self.normalize_image(background)

        # Simple masks with correct class_labels format (numbers to strings only)
        masks = {
            "gt_foreground": {
                "mask_data": (true_fg > 0).astype(np.uint8)*1,
                "class_labels": {0: "background", 1: "gt_foreground"}
            },
            "gt_boundary": {
                "mask_data": (true_bd > 0).astype(np.uint8)*2,
                "class_labels": {0: "background", 2: "gt_boundary"}
            },
            "pred_foreground": {
                "mask_data": (pred_fg > 0.5).astype(np.uint8)*3,
                "class_labels": {0: "background", 3: "pred_foreground"}
            },
            "pred_boundary": {
                "mask_data": (pred_bd > 0.5).astype(np.uint8)*4,
                "class_labels": {0: "background", 4: "pred_boundary"}
            }
        }

        return wandb.Image(bg_normalized, masks=masks, caption=title)

    def calculate_metrics(self, prediction, ground_truth):
        """Calculate dice and IoU metrics"""
        pred_binary = (prediction > 0.5).astype(int)
        true_binary = (ground_truth > 0).astype(int)

        intersection = np.sum(pred_binary * true_binary)
        pred_sum = np.sum(pred_binary)
        true_sum = np.sum(true_binary)

        # Dice score
        dice = 2.0 * intersection / (pred_sum + true_sum + 1e-7)

        # IoU score
        union = pred_sum + true_sum - intersection
        iou = intersection / (union + 1e-7)

        return float(dice), float(iou)

    def get_sample_category(self, target_volume):
        """Categorize sample by foreground density"""
        fg_ratio = np.sum(target_volume > 0) / target_volume.size

        if fg_ratio > 0.4:
            return "dense", fg_ratio
        elif fg_ratio < 0.1:
            return "sparse", fg_ratio
        else:
            return "medium", fg_ratio

    def setup_tracking_samples(self, val_loader, device):
        """Initialize fixed samples for progress tracking"""
        if self.initialized:
            return

        print("Setting up tracking samples...")
        val_iter = iter(val_loader)

        for i in range(2):  # Just 2 samples for simplicity
            try:
                x, y = next(val_iter)

                sample = {
                    'input': x[0, 0].cpu().numpy(),
                    'target_fg': y[0, 0].cpu().numpy(),
                    'target_bd': y[0, 1].cpu().numpy(),
                    'sample_id': f"sample_{i}",
                    'target_idx': None,
                }

                category, fg_ratio = self.get_sample_category(sample['target_fg'])
                sample['category'] = category
                sample['fg_ratio'] = fg_ratio

                self.tracking_samples.append(sample)
                print(f"  Sample {i}: {category} patch (fg_ratio: {fg_ratio:.3f})")

            except StopIteration:
                break

        self.initialized = True
        print("wandb will auto-assign colors to distinguish GT vs Pred masks")

    def log_progress_tracking(self, epoch, model, device):
        """Log same samples across epochs"""
        if not self.initialized:
            return

        model.eval()

        for sample in self.tracking_samples:
            try:
                # Get prediction
                x_tensor = torch.tensor(sample['input']).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(x_tensor)
                    pred_prob = torch.sigmoid(pred)

                # Extract middle slice
                #mid_z = sample['input'].shape[0] // 2
                #q_z = sample['input'].shape[0] // 4
                top_idx = sample['target_idx']
                if top_idx is None:
                    top_fg_ratio = 0
                    for i in range (2,12,2):
                        top_idx = sample['input'].shape[0] // i
                        fg_t = np.sum(sample['target_fg'][top_idx] > 0)
                        if fg_t > top_fg_ratio:
                            top_fg_ratio = fg_t
                            sample['target_idx'] = top_idx
                if top_idx == 0:
                    print ("Couldn't find a filled slice, using random 284")
                    top_idx = 248

                input_slice = sample['input'][top_idx]
                true_fg_slice = sample['target_fg'][top_idx]
                true_bd_slice = sample['target_bd'][top_idx]
                pred_fg_slice = pred_prob[0, 0, top_idx].cpu().numpy()
                pred_bd_slice = pred_prob[0, 1, top_idx].cpu().numpy()

                # Calculate metrics
                dice, iou = self.calculate_metrics(pred_fg_slice, true_fg_slice)

                # Create colored overlay
                title = f"Epoch {epoch} | {sample['category']} | Dice: {dice:.3f} | IoU: {iou:.3f}"
                overlay = self.create_colored_overlay(
                    input_slice, pred_fg_slice, pred_bd_slice,
                    true_fg_slice, true_bd_slice, title
                )
                row = [epoch, sample['sample_id'], sample['category'], dice, iou, sample['fg_ratio'], overlay]
                # Add to progress data
                self.progress_table.add_data(*row)
            except StopIteration:
                break
        wandb.log({"training_progress": self.progress_table})
        model.train()

    def log_validation_samples(self, epoch, model, val_loader, device):
        """Log different validation samples each epoch"""
        model.eval()
        val_iter = iter(val_loader)
        num_samples = self.log_config.get('num_samples_per_log', 2)

        for i in range(num_samples):
            try:
                x, y = next(val_iter)
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    pred = model(x)
                    pred_prob = torch.sigmoid(pred)

                # Extract middle slice from first sample in batch
                mid_z = x.shape[2] // 2
                input_slice = x[0, 0, mid_z].cpu().numpy()
                true_fg_slice = y[0, 0, mid_z].cpu().numpy()
                true_bd_slice = y[0, 1, mid_z].cpu().numpy()
                pred_fg_slice = pred_prob[0, 0, mid_z].cpu().numpy()
                pred_bd_slice = pred_prob[0, 1, mid_z].cpu().numpy()

                # Get sample info
                category, fg_ratio = self.get_sample_category(y[0, 0].cpu().numpy())
                dice, iou = self.calculate_metrics(pred_fg_slice, true_fg_slice)

                # Create overlay
                title = f"Epoch {epoch} | Val {i} | {category} | Dice: {dice:.3f} | IoU: {iou:.3f}"
                overlay = self.create_colored_overlay(
                    input_slice, pred_fg_slice, pred_bd_slice,
                    true_fg_slice, true_bd_slice, title
                )
                row = [epoch, f"val_{i}", category, dice, iou, fg_ratio, overlay]
                # Add to validation data
                self.validation_table.add_data(*row)

            except StopIteration:
                break
        wandb.log({"validation_progress": self.validation_table})
        model.train()

    def log_dataset_overview(self, val_loader):
        """One-time dataset overview"""
        if not self.log_config.get('log_dataset_overview', True):
            return

        print("Creating dataset overview...")
        val_iter = iter(val_loader)
        overview_data = []

        # Find one example of each category
        found_categories = set()
        attempts = 0

        while len(found_categories) < 3 and attempts < 10:
            try:
                x, y = next(val_iter)

                input_vol = x[0, 0].numpy()
                target_fg = y[0, 0].numpy()
                target_bd = y[0, 1].numpy()

                category, fg_ratio = self.get_sample_category(target_fg)

                if category not in found_categories:
                    mid_z = input_vol.shape[0] // 2

                    # Ground truth only overlay - wandb will assign colors
                    overlay = self.create_colored_overlay(
                        input_vol[mid_z],
                        pred_fg=np.zeros_like(target_fg[mid_z]),  # No prediction
                        pred_bd=np.zeros_like(target_bd[mid_z]),  # No prediction
                        true_fg=target_fg[mid_z],
                        true_bd=target_bd[mid_z],
                        title=f"{category.title()} sample"
                    )

                    overview_data.append([category, fg_ratio, overlay])
                    found_categories.add(category)

            except StopIteration:
                break

            attempts += 1

        if overview_data:
            overview_table = wandb.Table(
                columns=["category", "fg_ratio", "example"],
                data=overview_data
            )
            wandb.log({"dataset_overview": overview_table})


def add_image_logging(image_logger, epoch, model, val_loader , device, config):
    """Main logging function - much simpler!"""
    start_time = time.time()

    try:
        # Setup on first epoch
        if epoch == 0:
            image_logger.setup_tracking_samples(val_loader, device)
            image_logger.log_dataset_overview(val_loader)

        # Log progress tracking (same samples over time)
        prog_log_frequency = config.get('logging', {}).get('progress_log_frequency', 1)
        log_progress = (
                config.get('logging', {}).get('log_images', True) and
                (epoch % prog_log_frequency == 0 or epoch == 0)
        )
        if log_progress:
            image_logger.log_progress_tracking(epoch, model, device)
            elapsed = time.time() - start_time
            print(f"Progress image logging completed in {elapsed:.1f}s")
        else:
            print("Skipping progress image logging...")

        # Log validation samples (different each epoch)
        val_log_frequency = config.get('logging', {}).get('val_log_frequency', 1)
        log_validation = (
                config.get('logging', {}).get('log_images', True) and
                (epoch % val_log_frequency == 0 or epoch == 0)
        )
        if log_validation:
            image_logger.log_validation_samples(epoch, model, val_loader, device)
            elapsed = time.time() - start_time
            print(f"Validation image logging completed in {elapsed:.1f}s")
        else:
            print("Skipping validation image logging...")


    except Exception as e:
        print(f"Image logging failed: {e}")
        print("Training continues...")
