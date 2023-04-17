embs:
	chmod u+x slurm_jobs/create_bert_embs.slurm && chmod u+x slurm_jobs/create_ft_embs.slurm && chmod u+x slurm_jobs/create_glove_embs.slurm &&
	sbatch slurm_jobs/create_bert_embs.slurm && sbatch slurm_jobs/create_ft_embs.slurm && sbatch slurm_jobs/create_glove_embs.slurm

tr:
	chmod u+x slurm_jobs/train_lm.slurm && sbatch slurm_jobs/train_lm.slurm

te:
	chmod u+x slurm_jobs/test_lm.slurm && sbatch slurm_jobs/test_lm.slurm

pi:
	chmod u+x slurm_jobs/pickle_embeddings.slurm && sbatch slurm_jobs/pickle_embeddings.slurm


qm:
	squeue -u martijni

q:
	squeue

s:
	scancel $(id)

sa:
	scancel -u martijni

t:
	tail -f -n 1 $(name).out

gpu:
	nvidia-smi