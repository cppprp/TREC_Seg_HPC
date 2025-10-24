#!/usr/bin/env python3
"""
Batch Renormalization - Fixes tiling artifacts during inference
Based on: https://arxiv.org/abs/1702.03275
"""
import torch
import torch.nn as nn


class BatchRenorm3d(nn.Module):
    """Batch Renormalization for 3D data"""

    def __init__(self, num_features, eps=1e-5, momentum=0.01, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('rmax', torch.tensor(1.0))
        self.register_buffer('dmax', torch.tensor(0.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x shape: (N, C, D, H, W)
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean([0, 2, 3, 4])
            batch_var = x.var([0, 2, 3, 4], unbiased=False)

            with torch.no_grad():
                batch_std = torch.sqrt(batch_var + self.eps)
                running_std = torch.sqrt(self.running_var + self.eps)

                # Correction factors (clamped)
                r = (batch_std / running_std).clamp(1.0 / self.rmax, self.rmax)
                d = ((batch_mean - self.running_mean) / running_std).clamp(-self.dmax, self.dmax)

            # Normalize with correction
            x_norm = (x - batch_mean.view(1, -1, 1, 1, 1)) / batch_std.view(1, -1, 1, 1, 1)
            x_norm = x_norm * r.view(1, -1, 1, 1, 1) + d.view(1, -1, 1, 1, 1)

            # Update running statistics
            with torch.no_grad():
                self.running_mean += self.momentum * (batch_mean - self.running_mean)
                self.running_var += self.momentum * (batch_var - self.running_var)
                self.num_batches_tracked += 1
        else:
            # Inference: use global statistics (eliminates tiling artifacts!)
            x_norm = (x - self.running_mean.view(1, -1, 1, 1, 1)) / torch.sqrt(
                self.running_var.view(1, -1, 1, 1, 1) + self.eps)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)

        return x_norm

    def set_rmax_dmax(self, rmax, dmax):
        """Update the clipping bounds"""
        self.rmax.fill_(rmax)
        self.dmax.fill_(dmax)


class BatchRenormScheduler:
    """Gradually relaxes rmax and dmax during training"""

    def __init__(self, model, start_step=5000, rmax_step=40000, dmax_step=25000):
        self.model = model
        self.start_step = start_step
        self.rmax_step = rmax_step
        self.dmax_step = dmax_step
        self.current_step = 0

        # Find all BatchRenorm layers
        self.renorm_layers = [m for m in model.modules() if isinstance(m, BatchRenorm3d)]
        print(f"📊 Found {len(self.renorm_layers)} BatchRenorm3d layers")

    def step(self):
        """Call after each training batch"""
        self.current_step += 1

        if self.current_step < self.start_step:
            rmax, dmax = 1.0, 0.0
        else:
            # Linearly increase from start_step to target_step
            rmax_progress = min((self.current_step - self.start_step) / (self.rmax_step - self.start_step), 1.0)
            rmax = 1.0 + 2.0 * rmax_progress

            dmax_progress = min((self.current_step - self.start_step) / (self.dmax_step - self.start_step), 1.0)
            dmax = 5.0 * dmax_progress

        for layer in self.renorm_layers:
            layer.set_rmax_dmax(rmax, dmax)

    def get_current_bounds(self):
        """Get current rmax and dmax"""
        if self.renorm_layers:
            return {
                'rmax': self.renorm_layers[0].rmax.item(),
                'dmax': self.renorm_layers[0].dmax.item(),
                'step': self.current_step
            }
        return {'rmax': 1.0, 'dmax': 0.0, 'step': self.current_step}