import subprocess

max_len = [512, 256]

for l in max_len:
    # Slurm properties
    job_name = f"gs_tfidf_svm_{l}"
    out = f"out/svm_tfidf/w_gpu/gs_svm_tfidf_{l}_check_coefs.out" #f"out/gridsearch_lstm/{l}/gs_lstm_{emb}_{l}_{pad}.out"
    
    # Run sbatch
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=max_len={l} slurm_jobs/svm_tfidf/gridsearch_svm_tfidf.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())