import os
import subprocess
from pathlib import Path

def run(model, size, variation):
    # Slurm properties
    job_name = f"train_lm_classifier_{model}_{size}"
    out = f"out/train_lm_classifier/{model}_{size}.out"

    # Python properties
    dataset = f"{variation}_{size}"
    
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset} slurm_jobs/train_lm_classifier/job.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())


def run_all_combinations(models, variations, sizes):
    for model in models:
        for size in sizes:
            for variation in variations:
                run(model, size, variation)

def run_again(runs):
    for model, size, variation in runs:
        run(model, size, variation)

if __name__ == "__main__":
    
    # Run for all combinations
    models = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]
    variations = ["train_sliced_stair_twitter"]
    sizes = ["512", "256"]
    # run_all_combinations(models, variations, sizes)
    
    # Run again
    # runs = []
    runs = [["distilbert-base-uncased", "256", "train_sliced_stair_twitter"]]
    # run_again(runs)