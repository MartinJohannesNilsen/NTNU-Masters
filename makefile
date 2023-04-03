qm:
	squeue -u martijni

q:
	squeue
	
tr:
	chmod u+x train_lm.slurm && sbatch train_lm.slurm

te:
	chmod u+x test_lm.slurm && sbatch test_lm.slurm

s:
	scancel $(id)

sa:
	scancel -u martijni

t:
	tail -f -n 1 $(name).out

gpu:
	nvidia-smi