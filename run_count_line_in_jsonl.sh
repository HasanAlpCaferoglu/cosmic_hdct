#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=hdct_count_line_jsonl         #Setting a job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai                     # avg for lovelace_l40s | ai for ampere_a40
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1          # lovelace_l40s OR ampere_a40 (use with partition=ai and time:1-0:0:0)
#SBATCH --output=outputs/count_line_%j.out    # Standard output log file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alp.caferoglu@bilkent.edu.tr
#SBATCH --mem=32G

# ========================================
# ===== LOADING LIBRARY AND MODULES =====
# ========================================

module load java/16.0.2
source activate cosmic

echo "which python:"
which python
echo "-------"

echo "which -a python:"
which -a python
echo "-------"

echo "gcc version: "
which gcc
gcc --version
echo "-------"

echo "glibc version: "
ldd --version
echo "-------"

# Create the log directory if it doesn't exist
mkdir -p joblogs

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

# Print Cuda
echo "CUDA Version: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
echo "====================================================================================="
echo "====================================================================================="

export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .env

python3 -u count_line_in_jsonl.py