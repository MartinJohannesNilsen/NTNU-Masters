import subprocess

embeddings = ["glove", "glove_50", "fasttext", "bert"]
paddings = ["head", "tail", "split"]
lengths = ["256", "512"]

for emb in embeddings:
    for length in lengths:
        for pad_pos in paddings:
            # Slurm properties
            job_name = f"shooter_hold_out_create_{emb}_{length}_{pad_pos}"
            out = f"out/new_embs/shooter_hold_out_{emb}_{length}_{pad_pos}.out"
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb={emb},length={length},pad_pos={pad_pos} slurm_jobs/create_embs/job.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())