q:
	squeue -u martijni
	
r:
	chmod u+x job.slurm && sbatch job.slurm

z:
	chmod u+x zc.slurm && sbatch zc.slurm

g:
	chmod u+x git.slurm && sbatch git.slurm

s:
	scancel $(id)

sa:
	scancel -u martijni

t:
	tail -f -n 1 $(name).out

gpu:
	nvidia-smi