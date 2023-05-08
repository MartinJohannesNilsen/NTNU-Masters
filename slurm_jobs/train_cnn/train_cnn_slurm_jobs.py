import subprocess

emb_types = ["glove", "fasttext", "bert"]
pad_pos = ["head", "tail", "split"]

lengths = [256, 512]

emb_dict = {
    "bert": [768],
    "fasttext": [300],
    "glove": [50, 300]
}

for emb in emb_types:
        
    emb_dims = emb_dict[emb]

    for l in lengths:
        for pos in pad_pos:
            for dim in emb_dims:
                job_name = f"cnn_{emb}_{dim}_{l}_{pos}_train"
                sbatch_cmd = f"sbatch --export=emb={emb},dim={dim},pad_pos={pos},length={l} --job-name={job_name} --output=out/cnn_train/{job_name}.out slurm_jobs/train_cnn.slurm"

                subprocess.call(sbatch_cmd.split())


