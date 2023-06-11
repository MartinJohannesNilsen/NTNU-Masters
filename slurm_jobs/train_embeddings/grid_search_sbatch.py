import os
import subprocess
from pathlib import Path

embeddings = ["bert_768", "glove_300", "glove_50", "fasttext_300"]
models = ["svm", "xgboost", "gaussian", "nb", "knn"]
paddings = ["head", "tail", "split"]
variations = ["train_sliced_stair_twitter"]
sizes = ["512", "256"]

embeddings = ["bert_768"]
models = ["xgboost"]

for emb in embeddings:
    for model in models:
        for padding in paddings:
            for size in sizes:
                for variation in variations:
                    # Slurm properties
                    job_name = f"{model}grid_{emb}_{padding}_{size}"
                    out = f"out/grid_search_sklearn/{model}_{size}/{model}_{variation}_{emb}_{padding}_{size}.out"
                    memory = "500000" if model in ["svm", "xgboost"] and emb == "bert_768" and size == "512" else "200000" if emb == "bert_768" else "100000"

                    # Python properties
                    feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{variation}_{emb}_{padding}_{size}.h5")

                    # Run sbatch
                    sbatch_cmd = f"sbatch --mem={memory} --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model} slurm_jobs/train_embeddings/grid_search.slurm"
                    print(sbatch_cmd)
                    subprocess.call(sbatch_cmd.split())