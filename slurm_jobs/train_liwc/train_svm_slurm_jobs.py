from sys import subprocess

emb_types = ["glove", "fasttext", "bert"]

lengths = [256, 512]

for emb in emb_types:
    
    emb_str = ""
    sentence_length_str = ""
    emb_dim = 300

    for l in lengths:

        if emb == "glove":
            emb_dims = [50, 300]

            for dim in emb_dims:
                job_name = f"cnn_{emb}_{dim}_{l}_train"
                sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},weights={hash_value['weights']},path={path} --job-name={job_name} train_cnn.slurm"




# Define the command to run the sbatch job with the hash value
sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},weights={hash_value['weights']},start_epoch={hash_value['max_epoch']},path={path} --job-name={hash_value['model_hash']} --output=output/{hash_value['model_hash']}.out --constraint={hash_value['constraint']} zzz_slurm/job.slurm"
# print(sbatch_cmd)
# Submit the sbatch job using subprocess
subprocess.call(sbatch_cmd.split())