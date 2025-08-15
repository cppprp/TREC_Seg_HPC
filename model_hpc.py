#!/usr/bin/env python3
import sys
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
import torch
import numpy as np
import torch
from torch_em.model import UNet3d
from pathlib import Path
import json
from config_hpc import setup_config
import torch.nn.functional as F
# Improved loss function that handles class imbalance
class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, weights=[1.0, 2.0], smooth=1e-7):
        """
        Weighted Dice Loss for handling class imbalance

        Args:
            weights: List of weights for each class [foreground, boundary]
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.weights = weights
        self.smooth = smooth

    def forward(self, predictions, targets):
        # predictions: (B, C, D, H, W)
        # targets: (B, C, D, H, W)

        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)

        total_loss = 0
        for i in range(predictions.shape[1]):  # For each class
            pred_i = predictions[:, i]
            target_i = targets[:, i]

            # Flatten tensors
            pred_flat = pred_i.flatten()
            target_flat = target_i.flatten()

            # Calculate Dice coefficient
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice

            total_loss += self.weights[i] * dice_loss

        return total_loss / len(self.weights)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # Compute focal loss for each channel
        total_loss = 0
        for i in range(inputs.shape[1]):
            pred_i = inputs[:, i].flatten()
            target_i = targets[:, i].flatten()

            bce = F.binary_cross_entropy(pred_i, target_i, reduction='none')
            pt = torch.where(target_i == 1, pred_i, 1 - pred_i)
            focal_weight = self.alpha * (1 - pt) ** self.gamma

            focal_loss = focal_weight * bce
            total_loss += focal_loss.mean()

        return total_loss / inputs.shape[1]


class TverskyFocalLoss(torch.nn.Module):
    """Tversky loss with focal component.

        Args:
            alpha: False positive weight,
            beta: False negative weight,
            gamma: focal loss parameter
            smooth: smoothing factor
            """

    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-7):
        super().__init__()
        self.alpha = alpha  # Weight for false positives (lower = penalize FP more)
        self.beta = beta  # Weight for false negatives
        self.gamma = gamma  # Focal component
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        total_loss = 0
        for i in range(predictions.shape[1]):
            pred_i = predictions[:, i].flatten()
            target_i = targets[:, i].flatten()

            # Tversky components
            tp = (pred_i * target_i).sum()
            fp = (pred_i * (1 - target_i)).sum()
            fn = ((1 - pred_i) * target_i).sum()

            # Tversky index
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

            # Focal component
            tversky_loss = 1 - tversky
            focal_tversky = tversky_loss ** self.gamma

            total_loss += focal_tversky

        return total_loss / predictions.shape[1]


class FocalLossWithHardNegatives(torch.nn.Module):
    """Focal loss with hard negative mining for background false positives"""

    def __init__(self, alpha=0.25, gamma=2.0, hard_neg_ratio=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        # Standard focal loss
        total_loss = 0
        for i in range(predictions.shape[1]):
            pred_i = predictions[:, i].flatten()
            target_i = targets[:, i].flatten()

            # Binary cross entropy
            bce = F.binary_cross_entropy(pred_i, target_i, reduction='none')

            # Focal weights
            pt = torch.where(target_i == 1, pred_i, 1 - pred_i)
            focal_weight = self.alpha * (1 - pt) ** self.gamma

            focal_loss = (focal_weight * bce).mean()
            total_loss += focal_loss

        # Additional penalty for hard negatives (high confidence false positives)
        fg_channel = predictions[:, 0].flatten()  # Foreground channel
        bg_target = (targets[:, 0] == 0).flatten()  # Background pixels

        if bg_target.sum() > 0:
            # Find high confidence false positives
            false_pos_scores = fg_channel[bg_target]
            if len(false_pos_scores) > 10:  # Only if enough background pixels
                threshold = torch.quantile(false_pos_scores, 0.75)  # Top 25% hardest
                hard_negatives = false_pos_scores > threshold
                if hard_negatives.sum() > 0:
                    hard_neg_penalty = self.hard_neg_ratio * (false_pos_scores[hard_negatives] ** 2).mean()
                    total_loss += hard_neg_penalty

        return total_loss / predictions.shape[1]

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ComprehensiveMetrics:
    """Track multiple metrics during training"""

    def __init__(self, voxel_size_nm=650):
        self.voxel_size_nm = voxel_size_nm
        self.voxel_size_um = voxel_size_nm / 1000.0

    def compute_all_metrics(self, predictions, targets):
        """Compute all metrics for a batch"""
        # Apply sigmoid and threshold
        pred_probs = torch.sigmoid(predictions)
        pred_binary = (pred_probs[:, 0] > 0.5).float()  # Foreground channel
        target_binary = targets[:, 0].float()

        # Flatten for pixel-level metrics
        pred_flat = pred_binary.flatten().cpu().numpy()
        target_flat = target_binary.flatten().cpu().numpy()

        # Compute metrics
        metrics = {}

        # 1. Dice Score
        intersection = np.sum(pred_flat * target_flat)
        union = np.sum(pred_flat) + np.sum(target_flat)
        metrics['dice'] = 2.0 * intersection / (union + 1e-7)

        # 2. IoU
        intersection = np.sum(pred_flat * target_flat)
        union_iou = np.sum(pred_flat) + np.sum(target_flat) - intersection
        metrics['iou'] = intersection / (union_iou + 1e-7)

        # 3. Precision, Recall, F1
        if np.sum(pred_flat) > 0:
            metrics['precision'] = intersection / np.sum(pred_flat)
        else:
            metrics['precision'] = 0.0

        if np.sum(target_flat) > 0:
            metrics['recall'] = intersection / np.sum(target_flat)
        else:
            metrics['recall'] = 1.0 if np.sum(pred_flat) == 0 else 0.0

        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0

        return metrics
def load_trained_model(checkpoint_path, config_path=None, device='cuda'):
    """
    Load your trained plankton segmentation model

    Args:
        checkpoint_path: Path to your best_model.pth file
        config_path: Optional path to config.json (will try to find it automatically)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model in evaluation mode
    """

    print(f"Loading model from: {checkpoint_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model with same architecture as training
    model_config = config['model']
    print(f"Model config: {model_config}")

    model = UNet3d(
        in_channels=model_config['in_channels'],  # Should be 1
        out_channels=model_config['out_channels'],  # Should be 2 (fg + boundary)
        initial_features=model_config['initial_features'],  # Should be 32
        final_activation=model_config['final_activation']  # Should be None
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation dice: {checkpoint.get('val_dice', 'unknown')}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")

    # Move to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Verify model architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Input channels: {model_config['in_channels']}")
    print(f"Output channels: {model_config['out_channels']}")

    return model


def diagnose_logit_ranges(model, test_input):
    """Check how extreme your model's logits are"""
    with torch.no_grad():
        logits = model(test_input)

        print("🔍 Logit Analysis:")
        print(f"   Min logit: {logits.min().item():.2f}")
        print(f"   Max logit: {logits.max().item():.2f}")
        print(f"   Mean abs logit: {logits.abs().mean().item():.2f}")
        print(f"   Std logit: {logits.std().item():.2f}")

        # Check sigmoid distribution
        probs = torch.sigmoid(logits)
        binary_count = torch.sum((probs < 0.01) | (probs > 0.99))
        total_count = probs.numel()

        print(f"\n📊 After Sigmoid:")
        print(f"   Min prob: {probs.min().item():.6f}")
        print(f"   Max prob: {probs.max().item():.6f}")
        print(f"   Binary values (<0.01 or >0.99): {binary_count / total_count * 100:.1f}%")

        if logits.abs().mean() > 10:
            print("❌ EXTREME LOGITS - Model is severely overconfident!")
            return "extreme"
        elif logits.abs().mean() > 5:
            print("⚠️  High logits - Model is overconfident")
            return "high"
        else:
            print("✅ Normal logit range")
            return "normal"
