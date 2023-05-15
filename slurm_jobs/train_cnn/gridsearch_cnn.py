import subprocess

emb_type = ["glove", "glove_50", "fasttext", "bert"]
max_len = [512, 256]
pad_pos = ["head", "tail", "split"]

for emb in emb_type:
    for l in max_len:
        for pad in pad_pos:
            # Slurm properties
            job_name = f"gs_cnn_{emb}_{l}_{pad}"
            out = f"out/gridsearch_cnn/w_gpu/gs_cnn_{emb}_{l}_{pad}.out"
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb_type={emb},max_len={l},pad_pos={pad} slurm_jobs/train_cnn/gridsearch_cnn.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())