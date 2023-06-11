import subprocess

models = ["best_cnn_emb_liwc"]
slurm_paths = ["test_best_cnn_w_liwc"]


for model, path in zip(models, slurm_paths):
    # Slurm properties
    job_name = f"test_holdout_{model}"
    out = f"out/vote_holdout/{model}_vote_holdout.out"
    
    # Run sbatch
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} slurm_jobs/{path}.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())