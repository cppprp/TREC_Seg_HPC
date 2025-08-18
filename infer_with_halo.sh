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

# Configuration - UPDATE THESE VALUES
JOB_ID="35779375"                                    # Your training job ID
VOLUME_PATH="/scratch/asvetlove/inference_examples/POR_20to200_20231022_PM_01_epo_02/tomo"                # Volume to process
MASK_PATH="/scratch/asvetlove/inference_examples/Mask.tif"               # Optional mask
DATASET_NAME="POR_20to200_20231022_PM_01_epo_02"                    # Optional dataset name

# Optional processing parameters
START_SLICE="300"                                    # Starting slice (leave empty for full volume)
NUM_SLICES="470"                                     # Number of slices (leave empty for all)
NORMALIZE="--normalize"                           # Add --normalize flag or leave empty
NORM_MIN="-0.01"                                  # Normalization minimum
NORM_MAX="0.025"                                  # Normalization maximum

echo "🐍 Activating mamba environment..."
eval "$(mamba shell hook --shell bash)"
source /home/asvetlove/miniforge3/bin/activate
mamba activate trec_seg


# Create necessary directories
mkdir -p /scratch/$USER/logs
mkdir -p /scratch/$USER/tmp

# Set temporary directory to scratch for better I/O
export TMPDIR=/scratch/$USER/tmp

# Set CUDA device (SLURM handles GPU assignment)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Navigate to script directory
cd /home/asvetlove/TREC_Seg/code/

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

if [ ! -z "$NORMALIZE" ]; then
    CMD="$CMD $NORMALIZE --norm-min $NORM_MIN --norm-max $NORM_MAX"
fi

# Add HPC-specific optimizations
CMD="$CMD --num-threads $SLURM_CPUS_PER_TASK"
CMD="$CMD --verbose"
CMD="$CMD --log-file /scratch/$USER/logs/inference_${SLURM_JOB_ID}.log"

# Print command for debugging
echo "Executing command:"
echo "$CMD"
echo "================================================"

# Run inference
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Inference completed successfully!"

    # Optional: Copy results to permanent storage
    # rsync -av /scratch/$USER/inference_results/job_$JOB_ID/ /permanent/storage/path/

else
    echo "Inference failed with exit code $?"
    exit 1
fi

# Cleanup temporary files
rm -rf /scratch/$USER/tmp/*

echo "Job completed at $(date)"