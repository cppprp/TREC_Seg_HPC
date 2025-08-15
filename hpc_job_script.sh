#!/bin/bash
#SBATCH --job-name=plankton_hpc
#SBATCH --partition=gpu-el8
#SBATCH --constraint=gpu=A100
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=128789
#SBATCH --time=04:00:00
#SBATCH --output=logs/plankton_%j.out
#SBATCH --error=logs/plankton_%j.err

# Create logs directory
mkdir -p logs

echo "🚀 HPC Plankton Segmentation Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "User: $USER"
echo "Start time: $(date)"
echo "========================================"

echo "🐍 Activating mamba environment..."
eval "$(mamba shell hook --shell bash)"
source /home/asvetlove/miniforge3/bin/activate
mamba activate trec_seg

echo "✅ Environment activated:"
echo "   Python: $(which python)"
echo "   Python version: $(python --version)"

echo "📁 Changing to code directory..."
cd /home/asvetlove/TREC_Seg/code


export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=12

# Start training
echo ""
echo "🚀 STARTING HPC TRAINING"
echo "========================"
python train_flow_hpc.py 2>&1 | tee logs/training_${SLURM_JOB_ID}.log


echo ""
echo "========================"
echo "🏁 TRAINING COMPLETED"
echo "========================"
