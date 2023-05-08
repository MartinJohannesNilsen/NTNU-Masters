import os
import subprocess
from pathlib import Path

models = ["svm", "xgboost", "gaussian", "nb", "knn"]
embeddings = ["glove_300", "glove_50", "fasttext_300", "bert_768"]
paddings = ["head", "tail", "split"]
# variations = ["train_sliced_stair_twitter", "train_no_stair_twitter"]
# variations = ["train_sliced_stair_twitter"]
# variations = ["hold_out"]
sizes = ["512", "256"]

models = ["gaussian"]
embeddings = ["bert_768"]
# embeddings = ["glove_300", "fasttext_300", "bert_768", "glove_50"]
variations = ["train_sliced_stair_twitter"]
sizes = ["512"]

for model in models:
    for emb in embeddings:
        for padding in paddings:
            for size in sizes:
                for variation in variations:
                    # Slurm properties
                    job_name = f"{model}grid"
                    out = f"out/train_embeddings_grid_search_{model}/{model}_{variation}_{emb}_{padding}_{size}.out"

                    # Python properties
                    feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / f"{variation}_{emb}_{padding}_{size}.h5")
                    
                    # Run sbatch
                    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model} slurm_jobs/train_embeddings/grid_search.slurm"
                    print(sbatch_cmd)
                    subprocess.call(sbatch_cmd.split())