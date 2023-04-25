# Define the command to run the sbatch job with the hash value
sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},weights={hash_value['weights']},start_epoch={hash_value['max_epoch']},path={path} --job-name={hash_value['model_hash']} --output=output/{hash_value['model_hash']}.out --constraint={hash_value['constraint']} zzz_slurm/job.slurm"
# print(sbatch_cmd)
# Submit the sbatch job using subprocess
subprocess.call(sbatch_cmd.split())