import os
import subprocess
from pathlib import Path


# TODO Select train and test dataset
# train_datasets = ["train_sliced_stair_twitter", "train_no_stair_twitter"]
train_datasets = ["train_sliced_stair_twitter"]

models = ["svm", "nb", "knn", "xgboost", "gaussian"]
# models = ["knn"]
liwc_dictionaries = ["2022", "2015", "2007", "2001"]
# liwc_dictionaries = ["2022"] # , "2015", "2007", "2001"
for model in models:
    for liwc in liwc_dictionaries:
        for train_dataset in train_datasets:
            # Slurm properties
            job_name = f"train_liwc_{model}_{train_dataset}_{liwc}"
            out = f"out/train_liwc/{model}_{train_dataset}_{liwc}.out"

            # Python properties
            feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "h5" / liwc / f"{train_dataset}.h5")
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model} slurm_jobs/train_liwc/job.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())
