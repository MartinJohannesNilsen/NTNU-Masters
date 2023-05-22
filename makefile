slurm:
	chmod u+x slurm_jobs/$(dir)/$(name).slurm && sbatch slurm_jobs/$(dir)/$(name).slurm

tr:
	chmod u+x slurm_jobs/train_lm.slurm && sbatch slurm_jobs/train_lm.slurm

te:
	chmod u+x slurm_jobs/test_lm.slurm && sbatch slurm_jobs/test_lm.slurm

emb:
	chmod u+x slurm_jobs/create_bert_embs.slurm slurm_jobs/create_ft_embs.slurm slurm_jobs/create_glove_embs.slurm && sbatch slurm_jobs/create_bert_embs.slurm && sbatch slurm_jobs/create_ft_embs.slurm && sbatch slurm_jobs/create_glove_embs.slurm

qj:
	squeue -u olejlia

qjl:
	squeue -u olejlia --format="%i %P %.40j %u %t %M %D %R"

qm:
	squeue -u martijni

qml:
	squeue -u martijni --format="%i %P %.75j %u %t %M %D %R"

q:
	squeue

s:
	scancel $(id)

saj:
	scancel -u olejlia

sam:
	scancel -u martijni

t:
	tail -f -n 1 $(name).out

gpu:
	nvidia-smi

train-cnn:
	chmod u+x slurm_jobs/train_cnn.slurm  slurm_jobs/train_cnn1.slurm slurm_jobs/train_cnn2.slurm slurm_jobs/train_cnn3.slurm slurm_jobs/train_cnn4.slurm  && sbatch slurm_jobs/train_cnn.slurm && sbatch slurm_jobs/train_cnn1.slurm && sbatch slurm_jobs/train_cnn2.slurm && sbatch slurm_jobs/train_cnn3.slurm && sbatch slurm_jobs/train_cnn4.slurm