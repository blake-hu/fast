import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def extract_all_from_dir(results_base, embedding, task, dates=[]):
    results_dir = os.path.join(results_base, embedding, task)
    file_list = os.listdir(results_dir)

    results_df = pd.DataFrame()

    for filename in file_list:
        filepath = os.path.join(results_dir, filename)
        if not dates or any(date in filepath for date in dates):
            if "y_pred" in filepath:
                continue
            with open(filepath, 'r') as file:
                df_new = pd.read_csv(file)
                if results_df.empty:
                    results_df = df_new
                else:
                    results_df = pd.concat([results_df, df_new])

    return extract_hyperparameters(results_df, embedding, task) if not results_df.empty else {"error": "no csv found"}


def extract_from_file(filepath, embedding, task):

    with open(filepath, 'r') as file:
        results_df = pd.read_csv(file)

    return extract_hyperparameters(results_df, embedding, task)


def extract_hyperparameters(results_df, embedding, task):

    metric = "accuracy"
    second_metric = ""
    if task.lower() == "stsb":
        metric = "pearson"
        second_metric = "spearman"
    if task.lower() == "cola":
        metric = "mcc"
    if task.lower() in ["mrpc", "qqp"]:
        metric = "f1"
        second_metric = "accuracy"

    best = results_df[metric].max()
    best_row = results_df[results_df[metric] == best]

    hyperparam_grid = {}
    for key, val in best_row.to_dict().items():
        hyperparam_grid[key] = [list(val.values())[0]]
    print(f"embedding: {embedding}")
    print(f"task: {task}")
    if not second_metric:
        print(f"{metric}: {hyperparam_grid[metric][0]}")
    else:
        print(
            f"{metric} / {second_metric}: {hyperparam_grid[metric][0]} / {hyperparam_grid[second_metric][0]}")
    print()

    for key in ["pearson", "spearman", "accuracy", "mcc", "f1", "loss", "training time", "training energy", "embedding time", "embedding energy", "training time / epoch", "training energy / epoch", 'num_epochs.1']:
        if key in hyperparam_grid:
            del hyperparam_grid[key]

    if "num_classes" in hyperparam_grid:
        hyperparam_grid["num_classes"][0] = int(
            hyperparam_grid["num_classes"][0])

    return hyperparam_grid


def set_early_stop_epochs(param_grid):
    if "max_epochs" in param_grid and "num_epochs" in param_grid:
        param_grid["max_epochs"] = param_grid["num_epochs"]
        del param_grid["num_epochs"]

    elif "num_epochs" in param_grid:
        param_grid["max_epochs"] = param_grid["num_epochs"]

    param_grid["max_epochs"] = [int(param_grid["max_epochs"][0])]
    return param_grid
