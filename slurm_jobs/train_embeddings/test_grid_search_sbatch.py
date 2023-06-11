import os
import subprocess
from pathlib import Path

runs = [
    {"emb": "bert_768", "size": "512", "model": "svm", "pad_pos": "split"},
    {"emb": "bert_768", "size": "512", "model": "xgboost", "pad_pos": "head"},
    {"emb": "bert_768", "size": "512", "model": "xgboost", "pad_pos": "split"},
    {"emb": "bert_768", "size": "512", "model": "xgboost", "pad_pos": "tail"},
]
dataset = "train_sliced_stair_twitter"

for run in runs:
    emb = run["emb"]
    size = run["size"]
    model = run["model"]
    pad_pos = run["pad_pos"]

    # Slurm properties
    job_name = f"test{model}grid_{emb}_{pad_pos}_{size}"
    out = f"out/grid_search_sklearn_test/{model}_{size}/{model}_{dataset}_{emb}_{pad_pos}_{size}.out"
    memory = "200000" if model in ["svm", "xgboost"] and emb == "bert_768" and size == "512" else "200000" if emb == "bert_768" else "100000"

    # Python properties
    feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{dataset}_{emb}_{pad_pos}_{size}.h5")

    # Run sbatch
    sbatch_cmd = f"sbatch --mem={memory} --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model} slurm_jobs/train_embeddings/grid_search.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())