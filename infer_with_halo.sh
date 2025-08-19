#!/bin/bash
#SBATCH --job-name=plankton_hpc
#SBATCH --partition=gpu-el8
#SBATCH --constraint=gpu=A100
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=128789
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
set -o pipefail # should help with getting the correct return
# Configuration
JOB_ID="35779375"
VOLUME_PATH="/scratch/asvetlove/inference_examples/POR_20to200_20231022_PM_01_epo_02/tomo"
MASK_PATH="/scratch/asvetlove/inference_examples/Mask.tif"               # Optional mask
DATASET_NAME="POR_20to200_20231022_PM_01_epo_02"                    # Optional dataset name

# Optional processing parameters
START_SLICE="300"                                    # Starting slice (leave empty for full volume)
NUM_SLICES="470"                                     # Number of slices (leave empty for all)
NORM_MIN="-0.01"                                  # Normalization minimum
NORM_MAX="0.025"                                  # Normalization maximum

# Prediction parameters (adjust these based on your GPU memory)
BLOCK_SHAPE="128 128 128"                           # Block shape for prediction
HALO="32 32 32"                                     # Halo size
NUM_THREADS="8"                                     # Number of I/O threads

# Create logs directory
mkdir -p logs

echo "🚀 HPC Plankton Inference Job Started"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "User: $USER"
echo "Start time: $(date)"
echo "======================================"

echo "🐍 Activating mamba environment..."
eval "$(mamba shell hook --shell bash)"
source /home/asvetlove/miniforge3/bin/activate
mamba activate trec_seg

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "✅ Environment activated:"
echo "   Python: $(which python)"
echo "   Python version: $(python --version)"

echo "📁 Changing to code directory..."
cd /home/asvetlove/TREC_Seg/code

# Set threading environment (same as training)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=12

# Add GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Create necessary directories
mkdir -p /scratch/$USER/logs
mkdir -p /scratch/$USER/tmp

# Set temporary directory to scratch for better I/O
export TMPDIR=/scratch/$USER/tmp

# Verify input paths exist
if [ ! -e "$VOLUME_PATH" ]; then
    echo "❌ Error: Volume path not found: $VOLUME_PATH"
    exit 1
fi

if [ ! -z "$MASK_PATH" ] && [ ! -f "$MASK_PATH" ]; then
    echo "⚠️  Warning: Mask path not found: $MASK_PATH"
    echo "Proceeding without mask..."
    MASK_PATH=""
fi

# Build command with optional parameters
CMD="python inference_hpc.py $JOB_ID \"$VOLUME_PATH\""

if [ ! -z "$MASK_PATH" ]; then
    CMD="$CMD --mask-path \"$MASK_PATH\""
fi

if [ ! -z "$DATASET_NAME" ]; then
    CMD="$CMD --dataset-name \"$DATASET_NAME\""
fi

if [ ! -z "$START_SLICE" ]; then
    CMD="$CMD --start-slice $START_SLICE"
fi

if [ ! -z "$NUM_SLICES" ]; then
    CMD="$CMD --num-slices $NUM_SLICES"
fi

# Always apply normalization with custom min/max
CMD="$CMD --norm-min $NORM_MIN --norm-max $NORM_MAX"

# Add prediction parameters
CMD="$CMD --block-shape $BLOCK_SHAPE"
CMD="$CMD --halo $HALO"
CMD="$CMD --num-threads $NUM_THREADS"

echo ""
echo "📋 Configuration:"
echo "  Job ID: $JOB_ID"
echo "  Volume: $VOLUME_PATH"
echo "  Mask: ${MASK_PATH:-'None'}"
echo "  Dataset: $DATASET_NAME"
echo "  Start slice: ${START_SLICE:-'0'}"
echo "  Num slices: ${NUM_SLICES:-'All'}"
echo "  Normalization: [$NORM_MIN, $NORM_MAX]"
echo "  Block shape: $BLOCK_SHAPE"
echo "  Halo: $HALO"
echo "  I/O Threads: $NUM_THREADS"
echo "  OMP Threads: $OMP_NUM_THREADS"
echo ""
echo "🚀 STARTING INFERENCE"
echo "===================="
echo "Command: $CMD"
echo "===================="

# Run inference with logging (same pattern as training)
eval $CMD 2>&1 | tee logs/inference_${SLURM_JOB_ID}.log

# Check exit status
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "✅ Inference completed successfully!"

    # Show results location
    RESULTS_DIR="/scratch/$USER/inference_results/job_$JOB_ID/$DATASET_NAME"
    if [ -d "$RESULTS_DIR" ]; then
        echo "📁 Results saved to: $RESULTS_DIR"
        total_files=$(find "$RESULTS_DIR" -type f -name "*.tif" | wc -l)
        echo "📊 Output files: $total_files"

        # Show disk usage
        disk_usage=$(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
        echo "💾 Total size: $disk_usage"
    fi
else
    echo "❌ Inference failed with exit code $exit_code"
    echo "📋 Check log: logs/inference_${SLURM_JOB_ID}.log"
fi

echo "End time: $(date)"