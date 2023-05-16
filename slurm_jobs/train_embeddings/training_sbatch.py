import os
import subprocess
from pathlib import Path

models = ["svm", "nb", "knn", "xgboost", "gaussian"]
embeddings = ["glove_300", "glove_50", "fasttext_300", "bert_768"]
paddings = ["head", "tail", "split"]
variations = ["train_sliced_stair_twitter"]
sizes = ["512", "256"]

# models = ["nb"]
# embeddings = ["glove_50"]
# paddings = ["head"]
# # variations = ["train_sliced_stair_twitter"]
# sizes = ["256"]

for model in models:
    for emb in embeddings:
        for padding in paddings:
            for size in sizes:
                for variation in variations:
                    # Slurm properties
                    job_name = f"train_sklearn_emb_{model}_{variation}_{emb}_{padding}_{size}"
                    out = f"out/train_sklearn/embeddings/{model}_{size}/{model}_{variation}_{emb}_{padding}_{size}.out"

                    # Python properties
                    feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{variation}_{emb}_{padding}_{size}.h5")
                    test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{variation.replace('train', 'test')}_{emb}_{padding}_{size}.h5")
                    
                    # Run sbatch
                    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model},test_path={test_path} slurm_jobs/train_embeddings/training.slurm"
                    print(sbatch_cmd)
                    subprocess.call(sbatch_cmd.split())