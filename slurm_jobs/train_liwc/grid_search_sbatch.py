import os
import subprocess
from pathlib import Path

models = ["svm", "nb", "knn", "xgboost", "gaussian"]
train_datasets = ["train_sliced_stair_twitter_256_preprocessed", "train_sliced_stair_twitter_512_preprocessed"]
liwc_dictionaries = ["2022", "2015", "2007", "2001"]

for model in models:
    for liwc in liwc_dictionaries:
        for train_dataset in train_datasets:
            # Slurm properties
            size = "_256" if "256" in train_dataset else "_512" if "512" in train_dataset else ""
            job_name = f"{model}grid{size}_liwc_{liwc}"
            out = f"out/grid_search_sklearn/liwc/{model}{size}/{model}_{liwc}.out"

            # Python properties
            feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "preprocessed" / "splits" / "h5" / liwc / f"{train_dataset}.h5")
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model} slurm_jobs/train_liwc/grid_search.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())
