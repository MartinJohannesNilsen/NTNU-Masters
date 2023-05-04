import subprocess

# models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]
variations = ["train_sliced_stair_twitter", "train_no_stair_twitter"]
sizes = ["512", "256"]


models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]
# models = ["distilbert-base-uncased"]
# variations = ["train_sliced_stair_twitter"]
# sizes = ["512"]

for model in models:
    for size in sizes:
        for variation in variations:
            # Slurm properties
            job_name = f"test_{model}_{variation}_{size}"
            out = f"out/test_lm/{model}_{variation}_{size}.out"

            # Python properties
            dataset = f"{variation}{'_256' if size == '256' else ''}"
            checkpoint = "final"
            
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset},checkpoint={checkpoint} slurm_jobs/test_lm/job.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())

