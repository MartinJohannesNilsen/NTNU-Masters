#!/bin/sh
##SBATCH --partition=GPUQ
#SBATCH --partition=CPUQ
##SBATCH --partition=
#SBATCH --account=share-ie-idi
#SBATCH --time=96:00:00
#SBATCH --mem=150000
##SBATCH --gres=gpu:1
##SBATCH --constraint="P100|V100"
##SBATCH --nodes=1                          # 1 compute nodes
##SBATCH --ntasks-per-node=1                # 1 task per compute node
##SBATCH --cpus-per-task=2                  # 2 CPU cores
#SBATCH --mail-user=martijni@stud.ntnu.no
#SBATCH --mail-type=ALL
##SBATCH --export=model_path,test_path,output_path,threshold,list_of_thresholds
#SBATCH --export=model_path,test_path,output_path

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
cd "src/experiments/models/test"

# Run for loop of thresholds
#python test_sklearn.py $model_path $test_path --threshold $threshold --list-of-thresholds $list_of_thresholds --output $output_path
python test_sklearn.py $model_path $test_path --output $output_path

# Print idun stats
uname -a