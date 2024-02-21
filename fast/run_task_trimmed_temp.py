# %% [markdown]
# # FAST: Feedforward-Augmented Sentence Transformers

# %% [markdown]
# Notebook for running GLUE tasks.

# %% [markdown]
# # Setup

# %% [markdown]
# ## Modules

# %%
import argparse
import random
import time
import csv
import os
import pathlib
import itertools
from datetime import datetime
from collections import namedtuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from utils.feed_forward import FeedForward
from utils.cls import extract_cls_embeddings
from utils.mean_pooling import mean_pooling
from utils.energy import get_energy

# %%
# Standardized default seed
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Model name to generate embeddings')
parser.add_argument('--task', required=True, help='GLUE task')
parser.add_argument('--embedding', required=True, help='Embedding type')
parser.add_argument('--concat', required=True, help='Concatenation type')
parser.add_argument('--device', required=False,
                    help='Optional, specifies device for model to run')

args, unk = parser.parse_known_args()
model_param = args.model
task_param = args.task
embedding_param = args.embedding
concat_param = args.concat


# %%
device_name = "cpu"  # default device is CPU
if torch.cuda.is_available():
    if args.device is not None:
        device_name = args.device
    else:
        device_name = "cuda"  # CUDA for NVIDIA GPU
elif torch.backends.mps.is_available():
    # Metal Performance Shaders for Apple M-series GPU
    device_name = torch.device("mps")

device = torch.device(device_name)
print(device_name)

# %% [markdown]
# ## User parameters

# %% [markdown]
# Parameters to set:
# - Model
#     - MPNetBase
#     - DistilRoBERTaBase
#     - MPNetST
#     - DistilRoBERTaST
# - Task
#     - cola
#     - sst2
#     - mrpc
#     - stsb
#     - qqp
#     - mnli_matched
#     - mnli_mismatched
#     - qnli
#     - rte
#     - wnli
# - Embedding type
#     - cls
#     - mean_pooling
#     - sentence

# %% [markdown]
# ## Models

# %%
if model_param == "MPNetBase": # MPNet Base
    from transformers import MPNetTokenizer, MPNetModel
    tokenizer = MPNetTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = MPNetModel.from_pretrained("microsoft/mpnet-base").to(device)
elif model_param == "DistilRoBERTaBase": # DistilRoBERTa Base
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    model = RobertaModel.from_pretrained('distilroberta-base').to(device)
elif model_param == "MPNetST": # MPNet Sentence Transformer
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
elif model_param == "DistilRoBERTaST": # DistilRoBERTa Sentence Transformer
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to(device)
else:
    raise Exception(f"ERROR: Bad model_param")

# %%
# sentence_type: ["one", "two"]
# class_type: ["BC", "MC", "R"]
# input_size: int (represents input size of feedforward, aka embedding size)
# col_names: column names of relavent sentences on hugging face
# num_classes needs to be 1 for BC and R, only change value to number of classes for MC

TaskConfig = namedtuple("TaskConfig", ["sentence_type", "class_type", "num_classes", "input_size", "col_names"])

task_configs = {
    "cola": TaskConfig("one", "BC", 1, 768, ['sentence']),
    "sst2": TaskConfig("one", "BC", 1, 768, ['sentence']),
    "mrpc": TaskConfig("two", "BC", 1, 768, ['sentence1', 'sentence2']),
    "stsb": TaskConfig("two", "R", 1, 768, ['sentence1', 'sentence2']),
    "qqp": TaskConfig("two", "BC", 1, 768, ['question1', 'question2']),
    "mnli_matched": TaskConfig("two", "MC", 3, 768, ['premise', 'hypothesis']),
    "mnli_mismatched": TaskConfig("two", "MC", 3, 768, ['premise', 'hypothesis']),
    "qnli": TaskConfig("two", "BC", 1, 768, ['question', 'sentence']),
    "rte": TaskConfig("two", "BC", 1, 768, ['sentence1', 'sentence2']),
    "wnli": TaskConfig("two", "BC", 1, 768, ['sentence1', 'sentence2']),
}

task_config = task_configs[task_param]

# %% [markdown]
# ## Dataset

# %%
if task_param == "mnli_matched": 
    data = load_dataset("glue", "mnli") 
    val_key = "validation_matched"
    test_key = "test_matched"
elif task_param == "mnli_mismatched":
    data = load_dataset("glue", "mnli") 
    val_key = "validation_mismatched"
    test_key = "test_mismatched"
else:
    data = load_dataset("glue", task_param)
    val_key = "validation"
    test_key = "test"

data

# %% [markdown]
# ### Trimming data

# %%
N = 20000

train_N = min(N, len(data['train']))
train_idxs = np.random.choice(train_N, size=train_N, replace=False)
val_N = min(N, len(data['validation']))
val_idxs = np.random.choice(val_N, size=val_N, replace=False)
test_N = min(N, len(data['test']))
test_idxs = np.random.choice(test_N, size=test_N, replace=False)

# %% [markdown]
# # Embeddings

# %% [markdown]
# Labels come directly from dataset so no need to save to file

# %%
from sklearn.preprocessing import OneHotEncoder

Y_train = np.array(data["train"]["label"])
Y_val = np.array(data[val_key]["label"])
Y_test = np.array(data[test_key]["label"])

if task_config.class_type == "MC":
    Y_train = np.reshape(Y_train, (-1, 1))
    Y_val = np.reshape(Y_val, (-1, 1))
    Y_test = np.reshape(Y_test, (-1, 1))
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(Y_train)
    print(ohe.categories_)
     
    Y_train = ohe.transform(Y_train)
    Y_val = ohe.transform(Y_val)
    Y_test = ohe.transform(Y_test)

Y_train = Y_train[train_idxs]
Y_val = Y_val[val_idxs]
Y_test = Y_test[test_idxs]

print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)

# %% [markdown]
# ## Check for saved embeddings

# %%
cache_dir = f"./cache/{embedding_param}/{task_param}"
cache_path = pathlib.Path(cache_dir)
cache_path.mkdir(parents=True, exist_ok=True)

file_names = ['X_train', 'X_val', 'X_test']
paths = [pathlib.Path(cache_path / f"{f}_{model_param}.npy") for f in file_names]

use_cached_embeddings = False
if all(path.exists() for path in paths):
    print("Saved embeddings found!")
    X_train = np.load(paths[0])
    X_val = np.load(paths[1])
    X_test = np.load(paths[2])
    use_cached_embeddings = True
else:
    print("No saved embeddings found")

# %% [markdown]
# ## Generate embeddings

# %%
class GLUESingleSentence(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
class GLUEPairedSentence(Dataset):
    def __init__(self, texts1, texts2):
        self.texts1 = texts1
        self.texts2 = texts2

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]

        return text1, text2
    
def tokenize_batch(tokenizer, text_batch, max_len=512):
    encoded = tokenizer(text_batch,
                        add_special_tokens=True,
                        padding='max_length',
                        truncation=True,
                        max_length=max_len,
                        return_tensors='pt')
    return {k: v.to(device) for k, v in encoded.items()}

# %%
if task_config.sentence_type == "one":
    train_dataset = GLUESingleSentence(data['train']['sentence'])
    val_dataset = GLUESingleSentence(data['validation']['sentence'])
    test_dataset = GLUESingleSentence(data['test']['sentence'])
    
elif task_config.sentence_type == "two":
    key1, key2 = task_config.col_names
    
    train_dataset = GLUEPairedSentence(data['train'][key1], data['train'][key2])
    val_dataset = GLUEPairedSentence(data[val_key][key1], data[val_key][key2])
    test_dataset = GLUEPairedSentence(data[test_key][key1], data[test_key][key2])
        
else:
    raise Exception(f"{task_config.sentence_type}: sentence type not recognized")

# pick batch size based on GPU memory
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
def compute_single_embeddings(loader):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(loader):
            if embedding_param == "sentence":
                embed = model.encode(text)
            else:
                encoded = tokenize_batch(tokenizer, text)
                outputs = model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

                if embedding_param == "cls":
                    embed = extract_cls_embeddings(outputs)
                elif embedding_param == "mean_pooling":
                    embed = mean_pooling(outputs, encoded["attention_mask"])
                embed = embed.cpu().numpy()
                
            embeddings.append(embed)

    return np.concatenate(embeddings, axis=0)

def compute_pair_embeddings(loader):
    embeddings = []
    model.eval()
    with torch.no_grad():

        for text1, text2 in tqdm(loader):
            
            if embedding_param == "sentence":
                U = torch.tensor(model.encode(text1))
                V = torch.tensor(model.encode(text2))
            else: # cls or mean_pooling
                encoded1 = tokenize_batch(tokenizer, text1)
                encoded2 = tokenize_batch(tokenizer, text2)

                outputs1 = model(encoded1["input_ids"], attention_mask=encoded1["attention_mask"])
                outputs2 = model(encoded2["input_ids"], attention_mask=encoded2["attention_mask"])

                if embedding_param == "cls":
                    U = extract_cls_embeddings(outputs1)
                    V = extract_cls_embeddings(outputs2)
                elif embedding_param == "mean_pooling":
                    U = mean_pooling(outputs1, encoded1["attention_mask"])
                    V = mean_pooling(outputs2, encoded2["attention_mask"])

            embed = torch.cat([U, V], dim=1)
            embeddings.append(embed.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

if not use_cached_embeddings:
    start_time = time.time()

    if task_config.sentence_type == "one":
        X_train = compute_single_embeddings(train_loader)
        X_val = compute_single_embeddings(val_loader)
        X_test = compute_single_embeddings(test_loader)
    elif task_config.sentence_type == "two":
        X_train = compute_pair_embeddings(train_loader)
        X_val = compute_pair_embeddings(val_loader)
        X_test = compute_pair_embeddings(test_loader)
    
    embedding_time = time.time() - start_time
else:
    embedding_time = 0.0

# %%
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# %% [markdown]
# ## Saving embeddings to file

# %%
if not use_cached_embeddings:
    cache_path.mkdir(parents=True, exist_ok=True)

    file_names = ['X_train', 'X_val', 'X_test']
    paths = [pathlib.Path(cache_path / f"{f}_{model_param}.npy") for f in file_names]

    with open(paths[0], 'wb') as X_train_file:
        np.save(X_train_file, X_train)
    with open(paths[1], 'wb') as X_val_file:
        np.save(X_val_file, X_val)
    with open(paths[2], 'wb') as X_test_file:
        np.save(X_test_file, X_test)

# %% [markdown]
# ## Transforming Embeddings

# %%
# TRIMMING
X_train = X_train[train_idxs]
X_val = X_val[val_idxs]
X_test = X_test[test_idxs]

# %%
def compute_diff(embeddings):
    U = embeddings[:, :768]
    V = embeddings[:, 768:]
    return U - V

def compute_diff_abs(embeddings):
    return np.absolute(compute_diff(embeddings))

# %%
def compute_UVdiff(embeddings):
    return np.concatenate([embeddings, compute_diff(embeddings)], axis=1)

def compute_UVdiff_abs(embeddings):
    return np.concatenate([embeddings, compute_diff_abs(embeddings)], axis=1)

# %%
# transform_param = "diff"
transform_param = concat_param

# %%
EmbeddingTransformConfig = namedtuple("EmbeddingTransformConfig", ["input_size_factor", "transform"])
transform_config = {
    "UV": EmbeddingTransformConfig(2, lambda x: x),
    "diff": EmbeddingTransformConfig(1, compute_diff),
    "UVdiff": EmbeddingTransformConfig(3, compute_UVdiff),
    "diff_abs": EmbeddingTransformConfig(1, compute_diff_abs),
    "UVdiff_abs": EmbeddingTransformConfig(3, compute_UVdiff_abs),
}

# %%
# transform is only valid for two sentence tasks
if task_config.sentence_type == "two":
    f = transform_config[transform_param].transform
    X_train_computed = f(X_train)
    X_val_computed = f(X_val)
    X_test_computed = f(X_test)
else:
    X_train_computed = X_train
    X_val_computed = X_val
    X_test_computed = X_test

print(X_train_computed.shape)
print(X_val_computed.shape)
print(X_test_computed.shape)

# %% [markdown]
# # Training loop

# %%
input_size = task_config.input_size
if task_config.sentence_type == "two":
    input_size *= transform_config[transform_param].input_size_factor

param_grid = {
    'num_epochs': [50],
    'batch_size': [32, 512],
    'learning_rate': [1e-2, 1e-3],
    'category': [task_config.class_type],
    'norm': [False],
    'input_size': [input_size],
    'layer_size': [input_size // 4, input_size // 2, input_size, input_size * 2, input_size * 4],
    'num_classes': [task_config.num_classes],
    'num_layers': [1, 3, 5, 10],
    'weight_decay':[1e-2, 1e-4],
    'patience': [3],
    'min_delta': [0],
    'device': [device_name]
}

# Create a list of all combinations of hyperparameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
print(f"{len(all_params)} hyperparameter combinations")

# %%
# Setup for logging
console_output_filename = f'./output/{embedding_param}_{task_param}_console_output.txt'
with open(console_output_filename, 'a') as logfile:
    logfile.write('\n\nBEGIN TRAINING LOOP\n\n')

# Setup for saving results
results_folder = pathlib.Path(f"results/{embedding_param}/{task_param}")
results_folder.mkdir(parents=True, exist_ok=True)
save_file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_folder / f"val_{save_file_id}_{model_param}.csv"

# different metrics are recorded for classification vs regression tasks
if task_config.class_type in ["BC", "MC"]:
    metric_types = ['mcc', 'f1', 'accuracy']
elif task_config.class_type == "R":
    metric_types = ['pearson', 'spearman']

with open(results_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = metric_types + ['training time', 'embedding time', 'training energy', 'embedding energy'] + list(all_params[0].keys())
    writer.writerow(header)
print(f"saving results to ./{results_file}")
# Saves best accuracy for progress bar display
display_best = float("-inf")

# Iterate over all combinations of hyperparameters
bar = tqdm(enumerate(all_params), total=len(all_params))
for i, params in bar:
    # Formatting params to display
    print_params = params.copy()
    for param in ['category', 'device']:
        del print_params[param]

    # Initialize the model with current set of hyperparameters
    feed_forward = FeedForward(**params)

    metrics, train_times_per_epoch, energy_per_epoch = feed_forward.fit(X_train_computed,
                                                                        Y_train,
                                                                        X_val_computed,
                                                                        Y_val)
    
    # Log average training time per epoch for current parameter set
    # Note: FFN frequently stops early
    training_time = np.mean(train_times_per_epoch)
    training_energy = np.mean(energy_per_epoch) 
    # Compute energy for embedding generation
    embedding_energy = get_energy(embedding_time, device) # This method effectively just computes energy for a given time
    
    metric_vals = [metrics[mt] for mt in metric_types]
    
    # displaying results in progress bar
    display_recent = metrics["pearson" if task_config.class_type == "R" else "accuracy"]
    display_best = max(display_best, display_recent)
    bar.set_description(f"Best: {display_best:.5f}, Last: {display_recent:.5f}")

    # Write stats to log file
    with open(console_output_filename, 'a') as logfile:
        logfile.write(f"\n\nTraining with parameters:\n{print_params}")
        logfile.write(f"\nEarly stopped on epoch: {metrics['epoch']}")
        for name, val in zip(metric_types, metric_vals):
            logfile.write(f"\nValidation {name}: {val}")
        logfile.write(f"\nTraining time      : {training_time}") 
        logfile.write(f"\nEmbedding time      : {embedding_time}") 
        logfile.write(f"\nTraining energy    : {training_energy}") 
        logfile.write(f"\nEmbedding energy    : {embedding_energy}") 
    # Write to results csv
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row = metric_vals + [training_time, embedding_time, training_energy, embedding_energy] + list(params.values())
        writer.writerow(row)

