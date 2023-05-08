import subprocess

emb_type = ["glove", "glove_50", "fasttext"]
max_len = [256, 512]
pad_pos = ["head", "tail", "split"]

for emb in emb_type:
    for l in max_len:
        # Slurm properties
        job_name = f"gs_lstm_{emb}_{l}"
        out = f"out/train_cnn/gs_lstm_{emb}_{l}.out"
        
        # Run sbatch
        sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb_type={emb},max_len={l} slurm_jobs/train_lstm/gridsearch_lstm_glove_fasttext.slurm"
        print(sbatch_cmd)
        subprocess.call(sbatch_cmd.split())