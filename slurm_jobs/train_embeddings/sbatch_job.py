import os
import subprocess
from pathlib import Path

models = ["svm", "sgd", "nb", "knn", "xgboost", "gaussian"]
embeddings = ["glove", "glove_50", "fasttext", "bert"]
paddings = ["head", "tail", "split"]
# variations = ["train_sliced_stair_twitter", "train_no_stair_twitter"]
# variations = ["train_sliced_stair_twitter"]
# variations = ["hold_out"]
sizes = ["512", "256"]

embeddings = ["bert"]
paddings = ["tail"]
variations = ["train_sliced_stair_twitter"]
sizes = ["512"]

for model in models:
    for emb in embeddings:
        for padding in paddings:
            for size in sizes:
                for variation in variations:
                    # Slurm properties
                    job_name = f"train_embeddings_{model}_{variation}_{emb}_{padding}_{size}"
                    out = f"out/train_embeddings_griddy/{model}_{variation}_{emb}_{padding}_{size}.out"

                    # Python properties
                    feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / f"{variation}_{emb}_{padding}{'_256' if size == '256' else ''}.h5")
                    
                    # Run sbatch
                    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model} slurm_jobs/train_embeddings/job.slurm"
                    print(sbatch_cmd)
                    subprocess.call(sbatch_cmd.split())

