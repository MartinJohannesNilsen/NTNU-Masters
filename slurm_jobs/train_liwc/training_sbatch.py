import os
import subprocess
from pathlib import Path


# TODO Select train and test dataset
train_datasets = ["train_sliced_stair_twitter_256_preprocessed", "train_sliced_stair_twitter_512_preprocessed"]

models = ["svm", "nb", "knn", "xgboost", "gaussian"]
liwc_dictionaries = ["2022", "2015", "2007", "2001"]
for model in models:
    for liwc in liwc_dictionaries:
        for train_dataset in train_datasets:
            # Slurm properties
            size = "_256" if "256" in train_dataset else "_512" if "512" in train_dataset else ""
            job_name = f"train_sklearn_liwc_{model}_{train_dataset}_{liwc}"
            out = f"out/train_sklearn/liwc/{model}{size}/{model}_{liwc}.out"

            # Python properties
            feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "preprocessed" / "splits" / "h5" / liwc / f"{train_dataset}.h5")
            test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "preprocessed" / "splits" / "h5" / liwc / f"{train_dataset.replace('train', 'test')}.h5")
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model},test_path={test_path} slurm_jobs/train_liwc/training.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())
