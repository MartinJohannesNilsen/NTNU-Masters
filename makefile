embs:
	chmod u+x slurm_jobs/create_bert_embs.slurm && chmod u+x slurm_jobs/create_ft_embs.slurm && chmod u+x slurm_jobs/create_glove_embs.slurm &&
	sbatch slurm_jobs/create_bert_embs.slurm && sbatch slurm_jobs/create_ft_embs.slurm && sbatch slurm_jobs/create_glove_embs.slurm

tr:
	chmod u+x slurm_jobs/train_lm.slurm && sbatch slurm_jobs/train_lm.slurm

te:
	chmod u+x slurm_jobs/test_lm.slurm && sbatch slurm_jobs/test_lm.slurm

emb:
	chmod u+x slurm_jobs/create_bert_embs.slurm slurm_jobs/create_ft_embs.slurm slurm_jobs/create_glove_embs.slurm && sbatch slurm_jobs/create_bert_embs.slurm && sbatch slurm_jobs/create_ft_embs.slurm && sbatch slurm_jobs/create_glove_embs.slurm

qm:
	squeue -u olejlia

q:
	squeue

s:
	scancel $(id)

sa:
	scancel -u olejlia

t:
	tail -f -n 1 $(name).out

gpu:
	nvidia-smi