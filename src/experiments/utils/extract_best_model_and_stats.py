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
    else:
        output_path = cnn_out_path
    
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

if __name__ == "__main__":

    lstm_configs, lstm_results, lstm_emb_types = extract_configs_and_results_list(lstm_out_path)
    to_csv(lstm_emb_types, lstm_configs, lstm_results, "lstm")

    cnn_configs, cnn_results, cnn_emb_types = extract_configs_and_results_list(cnn_out_path)
    to_csv(cnn_emb_types, cnn_configs, cnn_results, "cnn")

    """ lstm_results = extract_configs_and_results(lstm_out_path)
    cnn_results = extract_configs_and_results(cnn_out_path)

    best_lstm_bert = get_best_result_for_emb_type(lstm_results, "bert")
    best_lstm_glove = get_best_result_for_emb_type(lstm_results, "glove")
    best_lstm_glove_50 = get_best_result_for_emb_type(lstm_results, "glove_50")
    best_lstm_fasttext = get_best_result_for_emb_type(lstm_results, "fasttext")

    best_cnn_bert = get_best_result_for_emb_type(cnn_results, "bert")
    best_cnn_glove = get_best_result_for_emb_type(cnn_results, "glove")
    best_cnn_glove_50 = get_best_result_for_emb_type(cnn_results, "glove_50")
    best_cnn_fasttext = get_best_result_for_emb_type(cnn_results, "fasttext")

    print(f"best lstm bert:\n{best_lstm_bert}")
    print(f"best lstm glove:\n{best_lstm_glove}")
    print(f"best lstm glove_50:\n{best_lstm_glove_50}")
    print(f"best lstm fasttext:\n{best_lstm_fasttext}")
    print(f"best cnn bert:\n{best_cnn_bert}")
    print(f"best cnn glove:\n{best_cnn_glove}")
    print(f"best cnn glove_50:\n{best_cnn_glove_50}")
    print(f"best cnn fasttext:\n{best_cnn_fasttext}") """

    """ cnn_configs, cnn_results, cnn_emb_types = extract_configs_and_results(cnn_out_path)

    print("CNN RESULTS AND CONFIGS")

    for emb_type, config, results in zip(cnn_emb_types, cnn_configs, cnn_results):
        print(f"emb: {emb_type}")
        print(f"config: {config}")
        print(f"results: {results}")

    cnn_f2_scores = [res["f2_score"] for res in cnn_results]
    best_cnn_f2_score = max(cnn_f2_scores)
    idx_best_cnn_f2_score = cnn_f2_scores.index(best_cnn_f2_score)
    print(f"Best: {best_cnn_f2_score} for {cnn_emb_types[idx_best_cnn_f2_score]} embeddings w config:\n{cnn_configs[idx_best_cnn_f2_score]}\nTotal results: {cnn_results[idx_best_cnn_f2_score]}")


    print("LSTM RESULTS AND CONFIGS")

    lstm_configs, lstm_results, lstm_emb_types = extract_configs_and_results(lstm_out_path)

    for emb_type, config, results in zip(lstm_emb_types, lstm_configs, lstm_results):
        print(f"emb: {emb_type}")
        print(f"config: {config}")
        print(f"results: {results}")

    lstm_f2_scores = [res["f2_score"] for res in lstm_results]
    best_lstm_f2_score = max(lstm_f2_scores)
    idx_best_lstm_f2_score = lstm_f2_scores.index(best_lstm_f2_score)

    print(f"Best: {best_lstm_f2_score} for {lstm_emb_types[idx_best_lstm_f2_score]} embeddings w config:\n{lstm_configs[idx_best_lstm_f2_score]}\nTotal results: {lstm_results[idx_best_lstm_f2_score]}")
 """
    """ cnn_configs, cnn_results, cnn_emb_types = extract_configs_and_results(cnn_out_path)
    
    cnn_f2_scores = [res["f2_score"] for res in cnn_results]
    best_cnn_f2_score = max(cnn_f2_scores)
    idx_best_cnn_f2_score = cnn_f2_scores.index(best_cnn_f2_score)

    print(cnn_f2_scores)
    print(f"Best: {best_cnn_f2_score} for {cnn_emb_types[idx_best_cnn_f2_score]} embeddings w config:\n{cnn_configs[idx_best_cnn_f2_score]}\nTotal results: {cnn_results[idx_best_cnn_f2_score]}")

    lstm_configs, lstm_results, lstm_emb_types = extract_configs_and_results(lstm_out_path)
    
    lstm_f2_scores = [res["f2_score"] for res in lstm_results]
    best_lstm_f2_score = max(lstm_f2_scores)
    idx_best_lstm_f2_score = lstm_f2_scores.index(best_lstm_f2_score)

    print(lstm_f2_scores)
    print(f"Best: {best_lstm_f2_score} for {lstm_emb_types[idx_best_lstm_f2_score]} embeddings w config:\n{lstm_configs[idx_best_lstm_f2_score]}\nTotal results: {lstm_results[idx_best_lstm_f2_score]}")
 """