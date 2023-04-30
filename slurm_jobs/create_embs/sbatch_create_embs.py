import subprocess

embeddings = ["glove", "glove_50", "fasttext", "bert"]
paddings = ["head", "tail", "split"]
purposes = ["train", "test", "hold_out"]
lengths = ["512", "256"]

for emb in embeddings:
    for length in lengths:
        for pad_pos in paddings:
            # Slurm properties
            job_name = f"create_{emb}_{length}"
            out = f"out/create_embs/{emb}_{length}.out"
            
            # Run sbatch
            sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb={emb},length={length},pad_pos={pad_pos} slurm_jobs/create_embs/job.slurm"
            print(sbatch_cmd)
            subprocess.call(sbatch_cmd.split())