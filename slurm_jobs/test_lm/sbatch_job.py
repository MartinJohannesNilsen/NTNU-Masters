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

def run(model, size, variation):
    # Slurm properties
    job_name = uniquify(f"test_{model}_{variation}_{size}")
    out = uniquify(f"out/test_lm_scores/{model}_{variation}_{size}.out")
    # out = f"out/test_lm/{model}_{variation}_{size}.out"

    # Python properties
    dataset = f"{variation}_{size}"
    test_file = f"{variation.replace('train', 'test')}_{size}"
    checkpoint = "final"
    
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset},test_file={test_file},checkpoint={checkpoint} slurm_jobs/test_lm/job.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())

def run_shooter_hold_out(model, size):
    for model in models:
        for size in sizes:
            # Slurm properties
            test_file = "shooter_hold_out"
            job_name = uniquify(f"test_{model}_{test_file}_{size}")
            out = uniquify(f"out/test_lm_scores/{model}_{test_file}_{size}.out")
            # out = f"out/test_lm/{model}_{variation}_{size}.out"

            # Python properties
            dataset = f"train_sliced_stair_twitter_{size}"
            test_file = f"{test_file}_{size}"
            checkpoint = "final"
            
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset},test_file={test_file},checkpoint={checkpoint} slurm_jobs/test_lm/job.slurm"
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
    # runs = [model, size, variation]
    # run_again(runs)
    
    # Run shooter hold out
    models = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]
    sizes = ["512", "256"]
    run_shooter_hold_out(models, sizes)