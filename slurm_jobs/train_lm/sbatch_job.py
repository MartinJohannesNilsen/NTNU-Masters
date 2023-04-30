import subprocess

models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]
variations = ["train_sliced_stair_twitter", "train_no_stair_twitter"]
sizes = ["512", "256"]

for model in models:
    for size in sizes:
        for variation in variations:
            # Slurm properties
            job_name = f"train_lm_{model}_{variation}_{size}"
            out = f"out/train_lm/{model}_{variation}_{size}.out"

            # Python properties
            dataset = f"{variation}_{'_256' if size == '256' else ''}"
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset} slurm_jobs/train_lm/job.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())

