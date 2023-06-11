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

def run(model, size, emb, padding):

    # Datasets
    train_dataset = "train_sliced_stair_twitter"
    # test_dataset = "test_sliced_stair_twitter"
    test_dataset = "shooter_hold_out"
    emb_with_dim = 'glove_50' if emb == "glove_50" else f"{emb}_{'768' if emb == 'bert' else '300'}"

    # Slurm properties
    job_name = uniquify(f"test_embeddings_{model}_{test_dataset}_{emb}_{padding}_{size}")
    out = uniquify(f"out/test_embeddings/{model}_{size}/{test_dataset}_{emb}_{padding}.out")

    # Python properties
    model_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "models" / "saved_models" / model / "embeddings" / emb / f"{train_dataset}_{emb_with_dim}_{padding}_{size}" / "sklearn_model.sav")
    # test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / "new" / f"{test_dataset}_{emb_with_dim}_{padding}_{size}.h5")
    test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / f"{test_dataset}_{emb_with_dim}_{padding}_{size}.h5")
    out_path = str(Path(os.path.abspath(__file__)).parents[2] / "out" / "test_embeddings" / f"{model}_{size}" / "posts" / f"{test_dataset}_{emb}_{padding}_posts.out")
    
    # Run sbatch
    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model_path={model_path},test_path={test_path},output_path={out_path} slurm_jobs/test_embeddings/job.slurm"
    print(sbatch_cmd)
    subprocess.call(sbatch_cmd.split())


def run_all_combinations(models, embeddings, paddings, sizes):
    for model in models:
        for emb in embeddings:
            for padding in paddings:
                for size in sizes:
                    run(model, size, emb, padding)

def run_again(runs):
    for model, size, emb, padding in runs:
        run(model, size, emb, padding)

if __name__ == "__main__":
    
    # Run for all combinations
    models = ["svm", "nb", "knn", "xgboost", "gaussian"]
    embeddings = ["glove", "glove_50", "fasttext", "bert"]
    paddings = ["head", "tail", "split"]
    sizes = ["512", "256"]
    # run_all_combinations(models, embeddings, paddings, sizes)
    
    # Run again
    # runs = []
    # runs = [["nb", "256", "glove_50", "head"]]
    runs = [["gaussian", "256", "bert", "head"]]
    run_again(runs)
