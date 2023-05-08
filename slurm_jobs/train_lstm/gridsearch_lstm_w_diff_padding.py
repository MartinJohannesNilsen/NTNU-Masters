import subprocess

emb_type = ["glove", "glove_50", "fasttext", "bert"]
max_len = [256, 512]
pad_pos = ["head", "split"]
check_padding = [True]

for emb in emb_type:
    for l in max_len:
        for pad in pad_pos:
            for should_check in check_padding:
                # Slurm properties
                job_name = f"gs_lstm_{emb}_{l}_{pad}_{should_check}_no_pack"
                out = f"out/gridsearch_lstm/no_pack_fix/gs_lstm_{emb}_{l}_{pad}_{should_check}.out"
                
                # Run sbatch
                sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb_type={emb},max_len={l},pad_pos={pad},check_padding={should_check} slurm_jobs/train_lstm/gridsearch_lstm_w_diff_padding.slurm"
                print(sbatch_cmd)
                subprocess.call(sbatch_cmd.split())