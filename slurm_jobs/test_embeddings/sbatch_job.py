import os
import subprocess
from pathlib import Path

models = ["svm", "nb", "knn", "xgboost", "gaussian"]
embeddings = ["glove", "glove_50", "fasttext", "bert"]
paddings = ["head", "tail", "split"]
variations = ["test_sliced_stair_twitter", "test_no_stair_twitter"]
# variations = ["hold_out"]
sizes = ["512", "256"]

for model in models:
    for emb in embeddings:
        for padding in paddings:
            for size in sizes:
                for variation in variations:
                    # Slurm properties
                    job_name = f"test_embeddings_{model}_{variation}_{emb}_{padding}_{size}"
                    out = f"out/test_embeddings/{model}_{variation}_{emb}_{padding}_{size}.out"

                    # Python properties
                    model_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "models" / "saved_models" / model / "embeddings" / emb / f"{variation}_{emb}_{padding}{'_256' if size == '256' else ''}" / "sklearn_model.sav")
                    test_path = str(Path(os.path.abspath(__file__)).parents[2] / "src" / "experiments" / "features" / "embeddings" / f"{variation}_{emb}_{padding}{'_256' if size == '256' else ''}.h5")
                    out_path = str(Path(os.path.abspath(__file__)).parents[2] / "out" / "test_liwc" / f"{model}_{variation}_{emb}_{padding}_{size}_posts.out")
                    
                    # Run sbatch
                    sbatch_cmd = f"sbatch --job-name={job_name} --output={out} --export=model_path={model_path},test_path={test_path},output_path={out_path} slurm_jobs/test_embeddings/job.slurm"
                    print(sbatch_cmd)
                    subprocess.call(sbatch_cmd.split())

