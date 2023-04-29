import os
import subprocess
from pathlib import Path


# TODO Select train and test dataset
# train_dataset = 
train_datasets = ["train_no_stair_twitter", "train_sliced_stair_twitter"]
test_datasets = ["test_sliced_stair_twitter", "test_no_stair_twitter"]
# test_datasets = ["shooter_hold_out_test"]


# models = ["svm", "knn", "xgboost", "gaussian", "nb" ]
models = ["svm"]
liwc_dictionaries = ["2022", "2015", "2007", "2001"]
for model in models:
    for liwc in liwc_dictionaries:
        for test_dataset in test_datasets:
            for train_dataset in train_datasets:
                # Slurm properties
                job_name = f"test_liwc_{train_dataset}_{model}_{test_dataset}_{liwc}"
                out = f"out/test_liwc_{train_dataset}/{model}_{test_dataset}_{liwc}.out"

                # Python properties
                model_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "models" / "saved_models" / model / "liwc" / liwc / f"{train_dataset}" / "sklearn_model.sav")
                test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "h5" / liwc / f"{test_dataset}.h5")
                out_path = str(Path(os.path.abspath(__file__)).parents[2] / "out" / f"test_liwc_{train_dataset}" / f"{model}_{test_dataset}_{liwc}_posts.out")
                
                # Run sbatch
                sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model_path={model_path},test_path={test_path},output_path={out_path} slurm_jobs/test_liwc/job.slurm"
                print(sbatch_cmd)
                subprocess.call(sbatch_cmd.split())
