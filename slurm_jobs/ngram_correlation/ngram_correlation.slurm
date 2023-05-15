#!/bin/sh
##SBATCH --partition=GPUQ
#SBATCH --partition=CPUQ
##SBATCH --partition=short
#SBATCH --account=share-ie-idi
#SBATCH --time=167:00:00
#SBATCH --mem=32000
##SBATCH --gres=gpu:1
##SBATCH --constraint="A100"
##SBATCH --nodes=1                          # 1 compute nodes
##SBATCH --ntasks-per-node=1                # 1 task per compute node
##SBATCH --cpus-per-task=2                  # 2 CPU cores
#SBATCH --mail-user=olejlia@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --export=max_len

WORKDIR=${SLURM_SUBMIT_DIR}

cd ${WORKDIR}
echo Information
echo ID: $SLURM_JOB_ID
echo Name: $SLURM_JOB_NAME
echo Directory: $SLURM_SUBMIT_DIR
echo Nodes: $SLURM_JOB_NODELIST
echo Number of nodes: $SLURM_JOB_NUM_NODES
echo Cores: $SLURM_CPUS_ON_NODE
echo Cores per node: $SLURM_CPUS_ON_NODE
echo Number of tasks per core: $SLURM_NTASKS
awk '/MemFree/ { printf "Allocated RAM: %.3fGB\n", $2/1024/1024 }' /proc/meminfo

# Load idun modules
module purge
module load Anaconda3/2020.07
module load Python/3.8.6-GCCcore-10.2.0

# Create environment and install requirements
echo "Creating conda environment"
conda create --force --name env
echo "Activating conda environment"
conda activate env
echo "Pip install requirements"
pip install -r requirements.txt --user -q
echo "Install nltk stopwords"
python -m nltk.downloader stopwords

# Run code
echo "Running code"
cd "src/experiments/models/train"
echo "Testing ngram correlation hehe eksd..."
python3 -u ngram_correlation.py --max_len $max_len

# Print idun stats
uname -a