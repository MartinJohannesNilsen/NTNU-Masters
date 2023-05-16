import subprocess

models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]
variation = "train_sliced_stair_twitter"
sizes = ["512", "256"]


# models = ["distilbert-base-uncased"]
# sizes = ["512"]

for model in models:
    for size in sizes:
            # Slurm properties
            job_name = f"gridsearch_lm_classifier_{model}_{size}"
            out = f"out/gridsearch_lm_classifier/{model}_{size}.out"

            # Python properties
            dataset = f"{variation}_{size}"
            
            # Run sbatch with constraint of a100 if large model
            # if size == '512' and model in ['bert-base-uncased', 'albert-base-v2']:
            # if size == '512':
                # sbatch_cmd = f'sbatch --job-name={job_name} --output={out} --constraint="A100" --export=model={model},size={size},dataset={dataset} slurm_jobs/train_lm/job.slurm'
            # else:
                # sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset} slurm_jobs/train_lm/job.slurm"
            
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model={model},size={size},dataset={dataset} slurm_jobs/gridsearch_lm_classifier/job.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())
