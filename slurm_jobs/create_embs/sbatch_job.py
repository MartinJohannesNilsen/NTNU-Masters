from sys import subprocess

embeddings = ["glove", "fasttext", "bert"]
paddings = ["head", "tail", "split"]
purposes = ["train", "test", "hold_out"]
sizes = ["512", "256"]

for emb in embeddings:
    for padding in paddings:
        for purpose in purposes:
            for size in sizes:
                job_name = f"create_emb_{purpose}_{emb}_{padding}_{size}"
                out = f"out/create_embs/{purpose}_{emb}_{padding}_{size}.out"
                # Define the command to run the sbatch job with the hash value
                sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=emb={emb},padding={padding},purpose={purpose},size={size} slurm_jobs/create_embs/job.slurm"
                print(sbatch_cmd)
                # Submit the sbatch job using subprocess
                subprocess.call(sbatch_cmd.split())


