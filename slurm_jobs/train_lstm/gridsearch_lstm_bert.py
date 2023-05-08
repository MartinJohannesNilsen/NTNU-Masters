import subprocess

max_len = [256, 512]

for l in max_len:
    # Slurm properties
    job_name = f"gs_lstm_bert_{l}"
    out = f"out/train_cnn/gs_lstm_bert_{l}.out"
    
    # Run sbatch
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb_type=bert,max_len={l} slurm_jobs/train_lstm/gridsearch_lstm_bert.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())