import os
import subprocess
from pathlib import Path


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        # path = filename + " (" + str(counter) + ")" + extension
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path

def run(model, liwc):

    # Datasets
    train_dataset = "train_sliced_stair_twitter_256_preprocessed"
    test_dataset = "shooter_hold_out_256"

    # Slurm properties
    job_name = uniquify(f"test_liwc_{train_dataset}_{model}_{test_dataset}_{liwc}")
    out = uniquify(f"out/test_liwc_{train_dataset}/{model}_{test_dataset}_{liwc}.out")

    # Python properties
    model_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "models" / "saved_models" / model / "liwc" / liwc / train_dataset / "sklearn_model.sav")
    # test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "preprocessed" / "splits" / "h5" / liwc / f"{test_dataset}.h5")
    test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "liwc" / "shooter_hold_out" / "h5" / liwc / f"{test_dataset}.h5")
    out_path = str(Path(os.path.abspath(__file__)).parents[2] / "out" / f"test_liwc_{train_dataset}" / f"{model}_{test_dataset}_{liwc}_posts.out")
    
    # Run sbatch
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model_path={model_path},test_path={test_path},output_path={out_path} slurm_jobs/test_liwc/job.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())


def run_all_combinations(models, liwc_dictionaries):
    for model in models:
        for liwc in liwc_dictionaries:
            run(model, liwc)

def run_again(runs):
    for model, liwc in runs:
        run(model, liwc)

if __name__ == "__main__":
    
    # Run for all combinations
    models = ["svm", "knn", "xgboost", "gaussian", "nb" ]
    liwc_dictionaries = ["2022", "2015", "2007", "2001"]
    # run_all_combinations(models, liwc)
    
    # Run again
    # runs = []
    runs = [["svm", "2022"]]
    run_again(runs)
