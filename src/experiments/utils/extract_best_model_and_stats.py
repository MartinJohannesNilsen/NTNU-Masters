import pandas as pd
import os
from pathlib import Path
import ast
from collections import defaultdict

lstm_out_path = Path(os.path.abspath(__file__)).parents[3] / "out" / "gridsearch_lstm" / "w_gpu"
cnn_out_path = Path(os.path.abspath(__file__)).parents[3] / "out" / "gridsearch_cnn" / "w_gpu"

res_sent = "Best results with config:"

# Read best from lstm

def extract_configs_and_results(output_path):
    res = defaultdict(lambda: defaultdict(list))

    for fpath in output_path.rglob("*.out"):
        print(fpath)
        file_components = str(Path(fpath).stem).split("_")
        emb_name = file_components[2] if file_components[3] != "50" else "_".join([file_components[2], file_components[3]])
        with open(fpath) as f:
            lines = f.readlines()
            
            i = -1
            line = lines[i].strip()
            while line != res_sent:
                i -= 1
                line = lines[i].strip()

            output_str = lines[i:]
            configs = ast.literal_eval(output_str[1])
            results = ast.literal_eval(output_str[2].split("Got metrics: ")[1])

            res[emb_name]["configs"].append(configs)
            res[emb_name]["results"].append(results)

    return res

def extract_configs_and_results_list(output_path):
    configs, results, emb_types = [], [], [] 

    for fpath in output_path.rglob("*.out"):
        print(fpath)
        file_components = str(Path(fpath).stem).split("_")
        emb_name = file_components[2] if file_components[3] != "50" else "_".join([file_components[2], file_components[3]])
        with open(fpath) as f:
            lines = f.readlines()
            
            i = -1
            line = lines[i].strip()
            while line != res_sent:
                i -= 1
                line = lines[i].strip()

            output_str = lines[i:]
            cfg = ast.literal_eval(output_str[1])
            res = ast.literal_eval(output_str[2].split("Got metrics: ")[1])

            configs.append(cfg)
            results.append(res)
            emb_types.append(emb_name)

    return configs, results, emb_types

def get_best_scoring_config(model_type, scoring):
    output_path = ""
    if model_type == "lstm":
        output_path = lstm_out_path
    elif model_type == "cnn":
        output_path = cnn_out_path
    else:
        output_path = Path(os.path.abspath(__file__)).parents[3] / "out" / "train_cnn_w_liwc"
    
    configs, scores, emb_types = extract_configs_and_results_list(output_path=output_path) 

    all_scores = [score[scoring] for score in scores]
    best_scoring = max(all_scores)
    best_score_idx = all_scores.index(best_scoring)

    return emb_types[best_score_idx], configs[best_score_idx], scores[best_score_idx]


def get_best_result_for_emb_type(res_dict: dict, emb_type: str):
    res_and_config = res_dict[emb_type]
    results = res_and_config["results"]
    configs = res_and_config["configs"]
    f2_scores = [res["f2_score"] for res in results]
    best_index = f2_scores.index(max(f2_scores))
    
    best_config_and_results = {
        "result": results[best_index],
        "config": configs[best_index]
    }

    return best_config_and_results


def to_csv(emb_types, configs, results, model_type: str):
    out_path = Path(os.path.abspath(__file__)).parents[1] / "models" / "results" / f"{model_type}_results.csv"
    print(str(out_path))

    max_lengths, padding_schemes, precision_scores, recall_scores, f1_scores, f2_scores = [], [], [], [], [], []
    for config, result in zip(configs, results):
        print(config)
        print(result)
        #precision_scores.append(result["precision"])
        max_lengths.append(config["max_len"])
        padding_schemes.append(config["pad_pos"])
        precision_scores.append(result["precision"])
        recall_scores.append(result["recall"])
        f1_scores.append(result["f1_score"])
        f2_scores.append(result["f2_score"])

    df = pd.DataFrame({
        "embedding_type": emb_types,
        "max_length": max_lengths,
        "padding": padding_schemes,
        "precision": precision_scores,
        "recall": recall_scores,
        "f1_score": f1_scores,
        "f2_score": f2_scores
    })

    df.to_csv(str(out_path), index=False)

def configs_to_csv(best_configs, model_type: str):
    out_path = Path(os.path.abspath(__file__)).parents[1] / "models" / "results" / f"{model_type}_configs.csv"

    configs = defaultdict(list)
    for config in best_configs:
        configs["emb_type"].append(config["emb_type"])
        configs["max_len"].append(config["max_len"])
        configs["pad_pos"].append(config["pad_pos"])
        configs["dropout"].append(config["dropout"])
        configs["lr"].append(config["lr"])
        configs["batch_size"].append(config["batch_size"])
        
        if model_type == "lstm":
            configs["hidden_size"].append(config["hidden_size"])
            configs["num_layers"].append(config["num_layers"])

    df = pd.DataFrame.from_dict(configs)
    

    df.to_csv(str(out_path), index=False)

if __name__ == "__main__":

    lstm_configs, _, _= extract_configs_and_results_list(lstm_out_path)
    configs_to_csv(lstm_configs, "lstm")

    cnn_configs, _, _ = extract_configs_and_results_list(cnn_out_path)
    configs_to_csv(cnn_configs, "cnn")