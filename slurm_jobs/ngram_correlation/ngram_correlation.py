import subprocess

max_len = [512, 256]

for l in max_len:
    # Slurm properties
    job_name = f"ngram_correlation_{l}"
    out = f"out/ngram_correlation/ngram_correlation_{l}_no_filter.out"
    
    # Run sbatch
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=max_len={l} slurm_jobs/ngram_correlation/ngram_correlation.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())