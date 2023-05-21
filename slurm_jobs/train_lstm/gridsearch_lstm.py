import subprocess

emb_type = ["bert"]
max_len = [512, 256]
pad_pos = ["head", "split", "tail"]

for emb in emb_type:
    for l in max_len:
        for pad in pad_pos:
            # Slurm properties
            job_name = f"gs_lstm_{emb}_{l}_{pad}"
            out = f"out/gridsearch_lstm/w_gpu/gs_lstm_{emb}_{l}_{pad}_eksd.out" #f"out/gridsearch_lstm/{l}/gs_lstm_{emb}_{l}_{pad}.out"
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb_type={emb},max_len={l},pad_pos={pad} slurm_jobs/train_lstm/gridsearch_lstm.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())