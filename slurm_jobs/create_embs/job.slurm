#!/bin/sh
##SBATCH --partition=GPUQ
#SBATCH --partition=CPUQ
##SBATCH --partition=short
#SBATCH --account=share-ie-idi
#SBATCH --time=140:00:00
#SBATCH --mem=32G
##SBATCH --gres=gpu:1
##SBATCH --constraint="P100|V100"
##SBATCH --nodes=1    
#SBATCH --job-name=name                      # 1 compute nodes
##SBATCH --ntasks-per-node=1                # 1 task per compute node
##SBATCH --cpus-per-task=2                  # 2 CPU cores
#SBATCH --export=emb,length,pad_pos

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
cd "src/experiments/utils"
python3 -u create_and_store_embs.py --emb $emb --length $length --pad_pos $pad_pos

# Print idun stats
uname -a