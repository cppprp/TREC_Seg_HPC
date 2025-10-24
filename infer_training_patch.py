import tifffile
from torch_em.util.prediction import predict_with_halo
import numpy as np
from inference_hpc import sigmoid_postprocess
from model_hpc import load_trained_model

import torch
def find_candidate_objects(raw_logits, threshold=0.5):
    """Find all potential objects in the prediction"""

    # Convert logits to probabilities
    probabilities = sigmoid_postprocess(raw_logits)

    # Create binary mask
    binary_mask = probabilities > threshold

    # Find connected components
    from skimage.measure import label
    labeled_objects = label(binary_mask)

    return labeled_objects, probabilities


def calculate_object_size_um(region, voxel_size_nm=650):
    """Calculate object size in micrometers"""
    voxel_size_um = voxel_size_nm / 1000  # Convert nm to µm

    # Volume-based size (most accurate for 3D)
    volume_voxels = region.area  # In skimage, 'area' is actually volume for 3D
    volume_um3 = volume_voxels * (voxel_size_um ** 3)

    # Approximate diameter assuming spherical
    diameter_um = 2 * ((3 * volume_um3) / (4 * np.pi)) ** (1 / 3)

    # Alternative: use major axis length
    major_axis_um = region.major_axis_length * voxel_size_um

    return {
        'diameter_um': diameter_um,
        'major_axis_um': major_axis_um,
        'volume_um3': volume_um3
    }


def validate_size_calculations(voxel_size_nm=650):
    """Check that size calculations make sense"""
    voxel_size_um = voxel_size_nm / 1000

    print(f"Voxel size: {voxel_size_um} µm")
    print(f"20µm = {20 / voxel_size_um:.1f} pixels")
    print(f"200µm = {200 / voxel_size_um:.1f} pixels")


# Should show: 20µm ≈ 31 pixels, 200µm ≈ 308 pixels
validate_size_calculations()



def extract_principal_axis_profile(raw_logits, region, buffer_pixels=5):
    """Extract logit profile along object's principal axis - CORRECTED"""

    # Get 3D bounding box and centroid
    min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
    centroid = region.centroid  # (z, y, x) order

    # For 3D objects, use the longest dimension
    z_extent = max_z - min_z
    y_extent = max_y - min_y
    x_extent = max_x - min_x

    # Find longest axis
    extents = [z_extent, y_extent, x_extent]
    longest_axis = np.argmax(extents)

    # Extract profile along longest axis through centroid
    center_z, center_y, center_x = [int(c) for c in centroid]

    if longest_axis == 0:  # Z-axis is longest
        start = max(0, min_z - buffer_pixels)
        end = min(raw_logits.shape[0], max_z + buffer_pixels)
        profile = raw_logits[start:end, center_y, center_x]
    elif longest_axis == 1:  # Y-axis is longest
        start = max(0, min_y - buffer_pixels)
        end = min(raw_logits.shape[1], max_y + buffer_pixels)
        profile = raw_logits[center_z, start:end, center_x]
    else:  # X-axis is longest
        start = max(0, min_x - buffer_pixels)
        end = min(raw_logits.shape[2], max_x + buffer_pixels)
        profile = raw_logits[center_z, center_y, start:end]

    return profile, longest_axis


def size_filter(region, min_size_um=20, max_size_um=200, voxel_size_nm=650):
    """Size filter using 3D bounding box - CORRECTED"""
    voxel_size_um = voxel_size_nm / 1000

    # Use bounding box dimensions
    min_z, min_y, min_x, max_z, max_y, max_x = region.bbox

    # Calculate size in micrometers
    size_z_um = (max_z - min_z) * voxel_size_um
    size_y_um = (max_y - min_y) * voxel_size_um
    size_x_um = (max_x - min_x) * voxel_size_um

    # Use maximum dimension as object size
    max_size = max(size_z_um, size_y_um, size_x_um)

    return min_size_um <= max_size <= max_size_um


def apply_comprehensive_filtering(raw_logits, voxel_size_nm=650):
    """CORRECTED version"""

    if isinstance(raw_logits, torch.Tensor):
        logits_np = raw_logits.cpu().numpy()
    else:
        logits_np = raw_logits

    # Find objects
    probabilities = 1 / (1 + np.exp(-logits_np))
    binary_mask = (probabilities > 0.5)

    from skimage.measure import label, regionprops
    labeled_objects = label(binary_mask)
    regions = regionprops(labeled_objects)

    valid_objects = []

    for region in regions:
        if region.area < 100:  # Skip tiny objects
            continue

        # Size filter
        if not size_filter(region, min_size_um=20, max_size_um=200, voxel_size_nm=voxel_size_nm):
            continue

        # Extract profile
        profile, axis = extract_principal_axis_profile(logits_np, region)

        if len(profile) < 5:  # Skip if profile too short
            continue

        # Phase analysis
        max_logit = np.max(profile)
        min_logit = np.min(profile)
        transition_sharpness = max_logit - min_logit

        # Simple thresholds
        if max_logit >= 5.0 and transition_sharpness >= 10.0:
            valid_objects.append(region.label)

    # Create output
    filtered_logits = np.zeros_like(logits_np)
    for obj_id in valid_objects:
        mask = (labeled_objects == obj_id)
        filtered_logits[mask] = logits_np[mask]

    print(f"Found {len(regions)} objects, kept {len(valid_objects)}")

    return filtered_logits, valid_objects

def calculate_phase_metrics(logit_profile):
    """Calculate metrics that distinguish true vs false objects"""

    # Convert to numpy for easier processing
    profile = logit_profile

    # Find peak and transition characteristics
    max_logit = np.max(profile)
    min_logit = np.min(profile)
    transition_sharpness = max_logit - min_logit

    # Find the steepest gradient (phase boundary signature)
    gradients = np.diff(profile)
    max_positive_gradient = np.max(gradients)
    max_negative_gradient = np.min(gradients)

    # Width of positive region
    positive_width = np.sum(profile > 0)
    total_width = len(profile)

    # Additional discriminative features
    logit_variance = np.var(profile[profile > 0]) if np.any(profile > 0) else 0

    return {
        'peak_strength': max_logit,  # True: ~9.4, False: ~6.4
        'transition_sharpness': transition_sharpness,  # True: ~17, False: ~13
        'max_gradient': max_positive_gradient,
        'positive_width': positive_width,
        'logit_variance': logit_variance
    }


def visualize_object_analysis(raw_logits, region, obj_id, output_dir="object_analysis",
                              voxel_size_nm=650, decision="unknown"):
    """Create detailed visualization for each analyzed object"""

    import os
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract profile and get analysis
    profile, axis = extract_principal_axis_profile(raw_logits, region)

    # Calculate all metrics
    max_logit = np.max(profile)
    min_logit = np.min(profile)
    transition_sharpness = max_logit - min_logit
    gradients = np.diff(profile)
    max_gradient = np.max(gradients) if len(gradients) > 0 else 0

    # Get object info
    centroid = region.centroid
    bbox = region.bbox
    voxel_size_um = voxel_size_nm / 1000

    # Calculate size
    size_z_um = (bbox[3] - bbox[0]) * voxel_size_um
    size_y_um = (bbox[4] - bbox[1]) * voxel_size_um
    size_x_um = (bbox[5] - bbox[2]) * voxel_size_um
    max_size_um = max(size_z_um, size_y_um, size_x_um)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: 2D slice with line overlay
    center_z = int(centroid[0])
    slice_2d = raw_logits[center_z, :, :]

    im1 = axes[0].imshow(slice_2d, cmap='gray')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Logit Value')

    # Draw the line path
    if axis == 0:  # Z-axis line (not visible in this slice)
        axes[0].plot(centroid[2], centroid[1], 'r+', markersize=10, markeredgewidth=2)
        axes[0].set_title(f'Object {obj_id} - Z-slice {center_z}\n(Line along Z-axis)')
    elif axis == 1:  # Y-axis line
        y_start = max(0, bbox[1] - 5)
        y_end = min(slice_2d.shape[0], bbox[4] + 5)
        axes[0].axhline(y=centroid[1], color='red', linewidth=2, linestyle='--',
                        xmin=y_start / slice_2d.shape[1], xmax=y_end / slice_2d.shape[1])
        axes[0].set_title(f'Object {obj_id} - Z-slice {center_z}\n(Red line shows profile path)')
    else:  # X-axis line
        x_start = max(0, bbox[2] - 5)
        x_end = min(slice_2d.shape[1], bbox[5] + 5)
        axes[0].axvline(x=centroid[2], color='red', linewidth=2, linestyle='--',
                        ymin=x_start / slice_2d.shape[0], ymax=x_end / slice_2d.shape[0])
        axes[0].set_title(f'Object {obj_id} - Z-slice {center_z}\n(Red line shows profile path)')

    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')

    # Panel 2: Line profile with analysis
    x_coords = np.arange(len(profile))
    axes[1].plot(x_coords, profile, 'b-', linewidth=2, label='Logit Profile')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Zero line')
    axes[1].axhline(y=max_logit, color='g', linestyle='--', alpha=0.7,
                    label=f'Peak: {max_logit:.1f}')
    axes[1].axhline(y=min_logit, color='r', linestyle='--', alpha=0.7,
                    label=f'Min: {min_logit:.1f}')

    # Mark steepest gradient
    if len(gradients) > 0:
        steepest_idx = np.argmax(gradients)
        axes[1].plot(steepest_idx, profile[steepest_idx], 'mo', markersize=8,
                     label=f'Max Gradient: {max_gradient:.1f}')

    axes[1].set_title(f'Logit Profile Analysis')
    axes[1].set_xlabel(f'Position along {"ZYX"[axis]}-axis (pixels)')
    axes[1].set_ylabel('Logit Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Metrics and decision
    axes[2].axis('off')

    # Object info
    info_text = f"""OBJECT {obj_id} ANALYSIS

SIZE METRICS:
- Volume: {region.area:,} voxels
- Dimensions: {size_x_um:.1f} × {size_y_um:.1f} × {size_z_um:.1f} μm
- Max size: {max_size_um:.1f} μm
- Size range: 20-200 μm {"✓" if 20 <= max_size_um <= 200 else "✗"}

PHASE METRICS:
- Peak Strength: {max_logit:.2f}
- Min Value: {min_logit:.2f}
- Transition Sharpness: {transition_sharpness:.2f}
- Max Gradient: {max_gradient:.2f}

FILTERING THRESHOLDS:
- Peak > 7.5: {"✓" if max_logit > 7.5 else "✗"}
- Transition > 15.0: {"✓" if transition_sharpness > 15.0 else "✗"}
- Gradient > 12.0: {"✓" if max_gradient > 12.0 else "✗"}"""

    axes[2].text(0.05, 0.95, info_text, transform=axes[2].transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Decision
    decision_color = "lightgreen" if decision == "KEPT" else "lightcoral"
    decision_text = f"DECISION: {decision}"
    axes[2].text(0.05, 0.05, decision_text, transform=axes[2].transAxes,
                 fontsize=14, weight='bold', verticalalignment='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.9))

    plt.tight_layout()

    # Save with meaningful filename
    filename = f"object_{obj_id:03d}_{decision.lower()}_{max_size_um:.1f}um.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def apply_comprehensive_filtering_with_viz(raw_logits, voxel_size_nm=650,
                                           output_dir="object_analysis"):
    """Filtering with comprehensive visualization of each object"""

    if isinstance(raw_logits, torch.Tensor):
        logits_np = raw_logits.cpu().numpy()
    else:
        logits_np = raw_logits

    # Find objects
    probabilities = 1 / (1 + np.exp(-logits_np))
    binary_mask = (probabilities > 0.5)

    from skimage.measure import label, regionprops
    labeled_objects = label(binary_mask)
    regions = regionprops(labeled_objects)

    valid_objects = []

    print(f"Analyzing {len(regions)} detected objects...")

    for region in regions:
        obj_id = region.label

        if region.area < 100:  # Skip tiny objects
            continue

        # Size filter
        size_passed = size_filter(region, min_size_um=20, max_size_um=200,
                                  voxel_size_nm=voxel_size_nm)

        if not size_passed:
            # Visualize rejected objects too
            visualize_object_analysis(logits_np, region, obj_id, output_dir,
                                      voxel_size_nm, "SIZE_REJECTED")
            continue

        # Extract profile and analyze
        profile, axis = extract_principal_axis_profile(logits_np, region)

        if len(profile) < 5:
            visualize_object_analysis(logits_np, region, obj_id, output_dir,
                                      voxel_size_nm, "PROFILE_TOO_SHORT")
            continue

        # Phase analysis
        max_logit = np.max(profile)
        min_logit = np.min(profile)
        transition_sharpness = max_logit - min_logit
        gradients = np.diff(profile)
        max_gradient = np.max(gradients) if len(gradients) > 0 else 0

        # Apply thresholds
        peak_threshold = 7.5
        sharpness_threshold = 15.0
        gradient_threshold = 12.0

        phase_passed = (max_logit >= peak_threshold and
                        transition_sharpness >= sharpness_threshold and
                        max_gradient >= gradient_threshold)

        if phase_passed:
            valid_objects.append(region.label)
            decision = "KEPT"
        else:
            decision = "PHASE_REJECTED"

        # Create visualization for this object
        viz_path = visualize_object_analysis(logits_np, region, obj_id, output_dir,
                                             voxel_size_nm, decision)

        print(f"Object {obj_id}: {decision} - visualization saved to {viz_path}")

    # Create filtered output
    filtered_logits = np.zeros_like(logits_np)
    for obj_id in valid_objects:
        mask = (labeled_objects == obj_id)
        filtered_logits[mask] = logits_np[mask]

    print(f"\nFinal results: {len(regions)} analyzed, {len(valid_objects)} kept")
    print(f"Visualizations saved in: {output_dir}/")

    return filtered_logits, valid_objects


def apply_comprehensive_filtering_with_mip(raw_logits, voxel_size_nm=650,
                                           output_dir="object_analysis_mip"):
    """Same filtering logic but with MIP visualizations"""

    if isinstance(raw_logits, torch.Tensor):
        logits_np = raw_logits.cpu().numpy()
    else:
        logits_np = raw_logits

    # Find objects
    probabilities = 1 / (1 + np.exp(-logits_np))
    binary_mask = (probabilities > 0.5)

    from skimage.measure import label, regionprops
    labeled_objects = label(binary_mask)
    regions = regionprops(labeled_objects)

    valid_objects = []

    print(f"Analyzing {len(regions)} detected objects with MIP views...")

    for region in regions:
        obj_id = region.label

        if region.area < 100:
            continue

        # Size filter
        size_passed = size_filter(region, min_size_um=20, max_size_um=200,
                                  voxel_size_nm=voxel_size_nm)

        if not size_passed:
            visualize_object_analysis_with_mip(logits_np, region, obj_id, output_dir,
                                               voxel_size_nm, "SIZE_REJECTED")
            continue

        # Phase analysis
        profile, axis = extract_principal_axis_profile(logits_np, region)

        if len(profile) < 5:
            visualize_object_analysis_with_mip(logits_np, region, obj_id, output_dir,
                                               voxel_size_nm, "PROFILE_TOO_SHORT")
            continue

        max_logit = np.max(profile)
        min_logit = np.min(profile)
        transition_sharpness = max_logit - min_logit
        gradients = np.diff(profile)
        max_gradient = np.max(gradients) if len(gradients) > 0 else 0

        # Apply thresholds (you can adjust these)
        peak_threshold = 7.5
        sharpness_threshold = 15.0
        gradient_threshold = 12.0

        phase_passed = (max_logit >= peak_threshold and
                        transition_sharpness >= sharpness_threshold and
                        max_gradient >= gradient_threshold)

        decision = "KEPT" if phase_passed else "PHASE_REJECTED"

        if phase_passed:
            valid_objects.append(region.label)

        # Create MIP visualization
        visualize_object_analysis_with_mip(logits_np, region, obj_id, output_dir,
                                           voxel_size_nm, decision)

        print(f"Object {obj_id}: {decision}")

    # Create filtered output
    filtered_logits = np.zeros_like(logits_np)
    for obj_id in valid_objects:
        mask = (labeled_objects == obj_id)
        filtered_logits[mask] = logits_np[mask]

    print(f"\nResults: {len(regions)} total, {len(valid_objects)} kept")
    print(f"MIP visualizations saved in: {output_dir}/")

    return filtered_logits, valid_objects


def visualize_object_analysis_with_mip(raw_logits, region, obj_id, output_dir="object_analysis",
                                       voxel_size_nm=650, decision="unknown"):
    """Enhanced visualization with Maximum Intensity Projections"""

    import os
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get the detected object mask and crop to bounding box
    probabilities = 1 / (1 + np.exp(-raw_logits))
    binary_mask = (probabilities > 0.5)

    # Create object-specific mask
    object_mask = np.zeros_like(binary_mask)
    coords = region.coords
    object_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    # Crop to object bounding box with some padding
    bbox = region.bbox
    pad = 5
    z_min, y_min, x_min = max(0, bbox[0] - pad), max(0, bbox[1] - pad), max(0, bbox[2] - pad)
    z_max, y_max, x_max = min(raw_logits.shape[0], bbox[3] + pad), min(raw_logits.shape[1], bbox[4] + pad), min(
        raw_logits.shape[2], bbox[5] + pad)

    # Cropped volumes
    cropped_logits = raw_logits[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_object = object_mask[z_min:z_max, y_min:y_max, x_min:x_max]

    # Extract profile and get analysis
    profile, axis = extract_principal_axis_profile(raw_logits, region)

    # Calculate metrics
    max_logit = np.max(profile)
    min_logit = np.min(profile)
    transition_sharpness = max_logit - min_logit
    gradients = np.diff(profile)
    max_gradient = np.max(gradients) if len(gradients) > 0 else 0

    # Get object info
    centroid = region.centroid
    voxel_size_um = voxel_size_nm / 1000

    # Calculate size
    size_z_um = (bbox[3] - bbox[0]) * voxel_size_um
    size_y_um = (bbox[4] - bbox[1]) * voxel_size_um
    size_x_um = (bbox[5] - bbox[2]) * voxel_size_um
    max_size_um = max(size_z_um, size_y_um, size_x_um)

    # Create figure with MIP views
    fig = plt.figure(figsize=(20, 12))

    # Create grid: 2 rows, 4 columns
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1.2])

    # Maximum Intensity Projections
    mip_xy = np.max(cropped_logits, axis=0)  # Project along Z
    mip_xz = np.max(cropped_logits, axis=1)  # Project along Y
    mip_yz = np.max(cropped_logits, axis=2)  # Project along X

    # Object projections for overlay
    obj_xy = np.max(cropped_object, axis=0)
    obj_xz = np.max(cropped_object, axis=1)
    obj_yz = np.max(cropped_object, axis=2)

    # Adjust centroid to cropped coordinates
    local_centroid = (centroid[0] - z_min, centroid[1] - y_min, centroid[2] - x_min)

    # Panel 1: XY projection (top view)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mip_xy, cmap='gray', origin='lower')
    # Overlay object in red
    red_overlay = np.zeros((*mip_xy.shape, 4))
    red_overlay[obj_xy > 0] = [1, 0, 0, 0.6]  # Semi-transparent red
    ax1.imshow(red_overlay, origin='lower')

    # Draw profile line
    if axis == 2:  # X-axis profile
        ax1.axvline(x=local_centroid[2], color='yellow', linewidth=2, linestyle='--')
        ax1.set_title(f'XY Projection (Top View)\nYellow line: profile path')
    elif axis == 1:  # Y-axis profile
        ax1.axhline(y=local_centroid[1], color='yellow', linewidth=2, linestyle='--')
        ax1.set_title(f'XY Projection (Top View)\nYellow line: profile path')
    else:
        ax1.plot(local_centroid[2], local_centroid[1], 'yellow', marker='x', markersize=10)
        ax1.set_title(f'XY Projection (Top View)\nYellow X: profile through Z')

    ax1.set_xlabel(f'X ({cropped_logits.shape[2]} px, {size_x_um:.1f}μm)')
    ax1.set_ylabel(f'Y ({cropped_logits.shape[1]} px, {size_y_um:.1f}μm)')

    # Panel 2: XZ projection (front view)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(mip_xz, cmap='gray', origin='lower', aspect='auto')
    red_overlay = np.zeros((*mip_xz.shape, 4))
    red_overlay[obj_xz > 0] = [1, 0, 0, 0.6]
    ax2.imshow(red_overlay, origin='lower', aspect='auto')

    if axis == 0:  # Z-axis profile
        ax2.axvline(x=local_centroid[0], color='yellow', linewidth=2, linestyle='--')
        ax2.set_title(f'XZ Projection (Front View)\nYellow line: profile path')
    elif axis == 2:  # X-axis profile
        ax2.axhline(y=local_centroid[2], color='yellow', linewidth=2, linestyle='--')
        ax2.set_title(f'XZ Projection (Front View)\nYellow line: profile path')
    else:
        ax2.plot(local_centroid[0], local_centroid[2], 'yellow', marker='x', markersize=10)
        ax2.set_title(f'XZ Projection (Front View)\nYellow X: profile through Y')

    ax2.set_xlabel(f'Z ({cropped_logits.shape[0]} px, {size_z_um:.1f}μm)')
    ax2.set_ylabel(f'X ({cropped_logits.shape[2]} px, {size_x_um:.1f}μm)')

    # Panel 3: YZ projection (side view)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(mip_yz, cmap='gray', origin='lower', aspect='auto')
    red_overlay = np.zeros((*mip_yz.shape, 4))
    red_overlay[obj_yz > 0] = [1, 0, 0, 0.6]
    ax3.imshow(red_overlay, origin='lower', aspect='auto')

    if axis == 0:  # Z-axis profile
        ax3.axvline(x=local_centroid[0], color='yellow', linewidth=2, linestyle='--')
        ax3.set_title(f'YZ Projection (Side View)\nYellow line: profile path')
    elif axis == 1:  # Y-axis profile
        ax3.axhline(y=local_centroid[1], color='yellow', linewidth=2, linestyle='--')
        ax3.set_title(f'YZ Projection (Side View)\nYellow line: profile path')
    else:
        ax3.plot(local_centroid[0], local_centroid[1], 'yellow', marker='x', markersize=10)
        ax3.set_title(f'YZ Projection (Side View)\nYellow X: profile through X')

    ax3.set_xlabel(f'Z ({cropped_logits.shape[0]} px, {size_z_um:.1f}μm)')
    ax3.set_ylabel(f'Y ({cropped_logits.shape[1]} px, {size_y_um:.1f}μm)')

    # Panel 4: Line profile analysis
    ax4 = fig.add_subplot(gs[0, 3])
    x_coords = np.arange(len(profile))
    ax4.plot(x_coords, profile, 'b-', linewidth=2, label='Logit Profile')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Zero line')
    ax4.axhline(y=max_logit, color='g', linestyle='--', alpha=0.7,
                label=f'Peak: {max_logit:.1f}')
    ax4.axhline(y=min_logit, color='r', linestyle='--', alpha=0.7,
                label=f'Min: {min_logit:.1f}')

    if len(gradients) > 0:
        steepest_idx = np.argmax(gradients)
        ax4.plot(steepest_idx, profile[steepest_idx], 'mo', markersize=8,
                 label=f'Max Gradient: {max_gradient:.1f}')

    ax4.set_title(f'Logit Profile Analysis')
    ax4.set_xlabel(f'Position along {"ZYX"[axis]}-axis (pixels)')
    ax4.set_ylabel('Logit Value')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: 3D shape summary
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.axis('off')

    shape_text = f"""OBJECT {obj_id} - 3D SHAPE ANALYSIS

DIMENSIONS (in micrometers):
- X-axis: {size_x_um:.1f} μm ({bbox[5] - bbox[2]} pixels)
- Y-axis: {size_y_um:.1f} μm ({bbox[4] - bbox[1]} pixels) 
- Z-axis: {size_z_um:.1f} μm ({bbox[3] - bbox[0]} pixels)

VOLUME:
- Total voxels: {region.area:,}
- Volume: {region.area * (voxel_size_um ** 3):.2f} μm³

PROFILE INFORMATION:
- Analysis along: {"Z" if axis == 0 else "Y" if axis == 1 else "X"}-axis (longest dimension)
- Profile length: {len(profile)} pixels ({len(profile) * voxel_size_um:.1f} μm)
- Axis chosen: {"ZYX"[axis]} ({["depth", "height", "width"][axis]})"""

    ax5.text(0.05, 0.95, shape_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Panel 6: Analysis results
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')

    results_text = f"""FILTERING ANALYSIS

PHASE METRICS:
- Peak Strength: {max_logit:.2f}
- Transition Sharpness: {transition_sharpness:.2f}
- Max Gradient: {max_gradient:.2f}

THRESHOLD CHECKS:
- Peak > 7.5: {"✓ PASS" if max_logit > 7.5 else "✗ FAIL"}
- Transition > 15.0: {"✓ PASS" if transition_sharpness > 15.0 else "✗ FAIL"}  
- Gradient > 12.0: {"✓ PASS" if max_gradient > 12.0 else "✗ FAIL"}
- Size 20-200 μm: {"✓ PASS" if 20 <= max_size_um <= 200 else "✗ FAIL"}"""

    ax6.text(0.05, 0.9, results_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    # Decision box
    decision_color = "lightgreen" if decision == "KEPT" else "lightcoral"
    decision_text = f"FINAL DECISION:\n{decision}"
    ax6.text(0.05, 0.25, decision_text, transform=ax6.transAxes,
             fontsize=14, weight='bold', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.4", facecolor=decision_color, alpha=0.9))

    plt.suptitle(f'Object {obj_id} Complete Analysis - MIP Views + Profile', fontsize=16)
    plt.tight_layout()

    # Save
    filename = f"object_{obj_id:03d}_{decision.lower()}_{max_size_um:.1f}um_mip.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath
def classify_object(metrics, thresholds):
    """Classify object as true plankton vs false positive"""

    # Your empirically derived thresholds
    peak_threshold = thresholds.get('peak_strength', 8.5)
    sharpness_threshold = thresholds.get('transition_sharpness', 15.0)
    gradient_threshold = thresholds.get('max_gradient', 16.0)

    # Classification logic
    is_valid = True

    # Must have strong peak confidence
    if metrics['peak_strength'] < peak_threshold:
        is_valid = False

    # Must have sharp phase transition
    if metrics['transition_sharpness'] < sharpness_threshold:
        is_valid = False

    # Must have steep gradient (sharp boundary)
    if metrics['max_gradient'] < gradient_threshold:
        is_valid = False

    return is_valid, metrics



model_path = "/mnt/duke-netapp/asvetlove/plankton_results/job_37227920/checkpoints/best_model.pth"
config_path = "/mnt/duke-netapp/asvetlove/plankton_results/job_37227920/logs/config.json"
volume_path = "/home/asvetlove/data/segmentation/inference_examples/POR_20to200_20231022_AM_01_epo_02/"
model = load_trained_model(model_path, config_path)
model.eval()
patch = tifffile.imread('/home/asvetlove/PycharmProjects/TREC_seg_unet/data/ml_patches/POR_20o200_20231022_AM_01_epo_01/patch_0002.tif')

prediction = predict_with_halo(
    patch,
    model,
    gpu_ids=[0],
    block_shape= [128,128,128],
    halo=[64,64,64],
    mask=None,
    postprocess=None,
    preprocess=None
)


foreground, boundaries = prediction[0], prediction[1]
#filtered_logits, kept_objects = apply_comprehensive_filtering(foreground, voxel_size_nm=650)
#filtered_logits, kept_objects = apply_comprehensive_filtering_with_viz(
#    foreground,
#    voxel_size_nm=650,
#    output_dir="object_analysis"
#)
filtered_logits, kept_objects = apply_comprehensive_filtering_with_mip(
    foreground,
    voxel_size_nm=650,
    output_dir="object_analysis_mip"
)
final_probabilities = sigmoid_postprocess(filtered_logits)

tifffile.imwrite('/home/asvetlove/data/segmentation/final_prob.tiff', final_probabilities)
#tifffile.imwrite('/home/asvetlove/data/segmentation/training_nosig_inff.tiff', foreground)