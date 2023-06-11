import os
import subprocess
from pathlib import Path

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        # path = filename + " (" + str(counter) + ")" + extension
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def run(emb, model, padding, size, variation):
    # Slurm properties
    # job_name = f"save_train_sklearn_emb_{model}_{variation}_{emb}_{padding}_{size}"
    # out = f"out/train_sklearn/embeddings_best/{model}_{size}/{model}_{variation}_{emb}_{padding}_{size}.out"
    job_name = uniquify(f"train_sklearn_emb_{model}_{variation}_{emb}_{padding}_{size}")
    out = uniquify(f"out/train_sklearn/embeddings/{model}_{size}/{model}_{variation}_{emb}_{padding}_{size}.out")
    # memory = "250000" if model in ["svm", "xgboost"] and emb == "bert_768" and size == "512" else "200000" if emb == "bert_768" or size == "512" else "100000"
    memory = "200000"

    # Python properties
    feature_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{variation}_{emb}_{padding}_{size}.h5")
    test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{variation.replace('train', 'test')}_{emb}_{padding}_{size}.h5")
    
    # Run sbatch
    sbatch_cmd = f"sbatch --mem={memory} --job-name={job_name} --output={out} --export=feature_path={feature_path},model={model},test_path={test_path} slurm_jobs/train_embeddings/training.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())


def run_all_combinations(embeddings, models, paddings, variations, sizes):
    for emb in embeddings:
        for model in models:
            for padding in paddings:
                for size in sizes:
                    for variation in variations:
                        run(emb, model, padding, size, variation)

def run_again(runs):
    for emb, model, padding, size, variation in runs:
        run(emb, model, padding, size, variation)

if __name__ == "__main__":
    
    # Run for all combinations
    embeddings = ["bert_768", "glove_300", "glove_50", "fasttext_300"]
    models = ["svm", "xgboost", "gaussian", "nb", "knn"]
    paddings = ["head", "tail", "split"]
    variations = ["train_sliced_stair_twitter"]
    sizes = ["512", "256"]
    # run_all_combinations(embeddings, models, paddings, variations, sizes)
    
    # Run oom killed again
    # runs = [['glove_50', 'svm', 'split', '256', 'train_sliced_stair_twitter']]
    # runs = [['bert_768', 'xgboost', 'tail', '256', 'train_sliced_stair_twitter']]
    # run_again(runs)