import random
import csv
import pathlib
import itertools
import argparse
from datetime import datetime
from collections import namedtuple
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from utils.feed_forward import FeedForward
from utils.cls import extract_cls_embeddings
from utils.mean_pooling import mean_pooling

# standardized default seed
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class GLUESingleSentence(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)


class GLUEPairedSentence(Dataset):
    def __init__(self, texts1, texts2, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts1 = texts1
        self.texts2 = texts2
        self.max_length = max_length

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]

        inputs1 = self.tokenizer.encode_plus(
            text1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs2 = self.tokenizer.encode_plus(
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return [inputs1['input_ids'].squeeze(0),
                inputs1['attention_mask'].squeeze(0),
                inputs2['input_ids'].squeeze(0),
                inputs2['attention_mask'].squeeze(0)]


class GLUEPairedSentenceST(Dataset):
    def __init__(self, texts1, texts2, max_length=512):
        self.texts1 = texts1
        self.texts2 = texts2
        self.max_length = max_length

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        inputs1 = self.texts1[idx]
        inputs2 = self.texts2[idx]

        return inputs1, inputs2


def tokenize_batch(tokenizer, text_batch):
    # print(text_batch)
    encoded = tokenizer(text_batch,
                        add_special_tokens=True,
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors='pt')
    # print(encoded)
    return encoded


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    mean_embeddings = torch.sum(
        masked_embeddings, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)


def extract_cls_embeddings(model_output):
    last_hidden_state = model_output['last_hidden_state']
    cls_embedding = last_hidden_state[:, 0, :]
    return cls_embedding


def compute_single_embeddings(loader):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            if embedding_param == "cls":
                embed = extract_cls_embeddings(outputs)
            elif embedding_param == "mean_pooling":
                embed = mean_pooling(outputs, attention_mask)
            elif embedding_param == "sentence":
                embed = torch.tensor(model.encode(input_ids))

            embeddings.append(embed.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def compute_pair_embeddings(loader):
    embeddings = []
    model.eval()
    with torch.no_grad():

        if embedding_param in ["cls", "mean_pooling"]:
            for input_ids1, attention_mask1, input_ids2, attention_mask2 in tqdm(loader):
                input_ids1 = input_ids1.to(device)
                attention_mask1 = attention_mask1.to(device)
                input_ids2 = input_ids2.to(device)
                attention_mask2 = attention_mask2.to(device)

                outputs1 = model(input_ids1, attention_mask=attention_mask1)
                outputs2 = model(input_ids2, attention_mask=attention_mask2)

                if embedding_param == "cls":
                    U = extract_cls_embeddings(outputs1)
                    V = extract_cls_embeddings(outputs2)
                elif embedding_param == "mean_pooling":
                    U = mean_pooling(outputs1, attention_mask1)
                    V = mean_pooling(outputs2, attention_mask2)

                embed = torch.cat([U, V], dim=1)
                embeddings.append(embed.cpu().numpy())

        elif embedding_param == "sentence":
            for input_ids1, input_ids2 in tqdm(loader):
                U = torch.tensor(model.encode(input_ids1))
                V = torch.tensor(model.encode(input_ids2))
                embed = torch.cat([U, V], dim=1)
                embeddings.append(embed.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Model name to generate embeddings')
    parser.add_argument('--task', required=True, help='GLUE task')
    parser.add_argument('--embedding', required=True, help='Embedding type')
    parser.add_argument('--device', required=False,
                        help='Optional, can specific specific device for model to run')

    args, unk = parser.parse_known_args()
    model_param = args.model
    task_param = args.task
    embedding_param = args.embedding

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
    print("Device:", device_name)

    # selecting model
    if model_param == "MPNetBase":  # MPNet Base
        from transformers import MPNetTokenizer, MPNetModel
        tokenizer = MPNetTokenizer.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2')
        model = MPNetModel.from_pretrained("microsoft/mpnet-base").to(device)
    elif model_param == "DistilRoBERTaBase":  # DistilRoBERTa Base
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
        model = RobertaModel.from_pretrained('distilroberta-base').to(device)
    elif model_param == "MPNetST":  # MPNet Sentence Transformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2').to(device)
    elif model_param == "DistilRoBERTaST":  # DistilRoBERTa Sentence Transformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            'sentence-transformers/all-distilroberta-v1').to(device)
    else:
        raise Exception(f"ERROR: Bad model_param")

    TaskConfig = namedtuple(
        "TaskConfig", ["sentence_type", "class_type", "input_size", "col_names"])
    task_configs = {
        "cola": TaskConfig("one", "BC", 768, ['sentence']),
        "sst2": TaskConfig("one", "BC", 768, ['sentence']),
        "mrpc": TaskConfig("two", "BC", 768*2, ['sentence1', 'sentence2']),
        "stsb": TaskConfig("two", "R", 768*2, ['sentence1', 'sentence2']),
        "qqp": TaskConfig("two", "BC", 768*2, ['question1', 'question2']),
        "mnli_matched": TaskConfig("two", "MC", 768*2, ['premise', 'hypothesis']),
        "mnli_mismatched": TaskConfig("two", "MC", 768*2, ['premise', 'hypothesis']),
        "qnli": TaskConfig("two", "BC", 768*2, ['question', 'sentence']),
        "rte": TaskConfig("two", "BC", 768*2, ['sentence1', 'sentence2']),
        "wnli": TaskConfig("two", "BC", 768*2, ['sentence1', 'sentence2']),
    }

    task_config = task_configs[task_param]

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

    Y_train = np.array(data["train"]["label"])
    Y_val = np.array(data[val_key]["label"])
    Y_test = np.array(data[test_key]["label"])

    # checking for saved embeddings
    cache_dir = f"./cache/{embedding_param}/{task_param}"
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    file_names = ['X_train', 'X_val', 'X_test']
    paths = [pathlib.Path(
        cache_path / f"{f}_{model_param}.npy") for f in file_names]

    if all(path.exists() for path in paths):
        print("saved embeddings found!")
        X_train = np.load(paths[0])
        X_val = np.load(paths[1])
        X_test = np.load(paths[2])
    else:
        print("no saved embeddings found")
        if task_config.sentence_type == "one":
            train_dataset = GLUESingleSentence(
                data['train']['sentence'], tokenizer)
            val_dataset = GLUESingleSentence(
                data['validation']['sentence'], tokenizer)
            test_dataset = GLUESingleSentence(
                data['test']['sentence'], tokenizer)

        elif task_config.sentence_type == "two":
            key1 = task_config.col_names[0]
            key2 = task_config.col_names[1]

            if embedding_param in ["cls", "mean_pooling"]:
                train_dataset = GLUEPairedSentence(
                    data['train'][key1], data['train'][key2], tokenizer)
                val_dataset = GLUEPairedSentence(
                    data[val_key][key1], data[val_key][key2], tokenizer)
                test_dataset = GLUEPairedSentence(
                    data[test_key][key1], data[test_key][key2], tokenizer)
            elif embedding_param == "sentence":
                train_dataset = GLUEPairedSentenceST(
                    data['train'][key1], data['train'][key2])
                val_dataset = GLUEPairedSentenceST(
                    data[val_key][key1], data[val_key][key2])
                test_dataset = GLUEPairedSentenceST(
                    data[test_key][key1], data[test_key][key2])

        else:
            raise Exception(
                f"{task_config.sentence_type}: sentence type not recognized")

        # pick batch size based on GPU memory
        batch_size = 16
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        if task_config.sentence_type == "one":
            train_embed = compute_single_embeddings(train_loader)
            val_embed = compute_single_embeddings(val_loader)
            test_embed = compute_single_embeddings(test_loader)
        elif task_config.sentence_type == "two":
            train_embed = compute_pair_embeddings(train_loader)
            val_embed = compute_pair_embeddings(val_loader)
            test_embed = compute_pair_embeddings(test_loader)

        X_train = train_embed
        X_val = val_embed
        X_test = test_embed

        # saving embeddings
        cache_path.mkdir(parents=True, exist_ok=True)

        file_names = ['X_train', 'X_val', 'X_test']
        paths = [pathlib.Path(
            cache_path / f"{f}_{model_param}.npy") for f in file_names]

        with open(paths[0], 'wb') as X_train_file:
            np.save(X_train_file, X_train)
        with open(paths[1], 'wb') as X_val_file:
            np.save(X_val_file, X_val)
        with open(paths[2], 'wb') as X_test_file:
            np.save(X_test_file, X_test)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    param_grid = {
        'num_epochs': [50],
        'batch_size': [32, 128, 512],
        'learning_rate': [1e-2, 1e-3],
        'category': [task_config.class_type],
        'norm': [False],
        'input_size': [task_config.input_size],
        'layer_size': [task_config.input_size],
        'num_layers': [1, 2, 3],
        'weight_decay': [1e-2, 1e-3, 1e-4],
        'patience': [3],
        'min_delta': [0],
        'device': [device_name]
    }

    # Create a list of all combinations of hyperparameters
    all_params = [dict(zip(param_grid.keys(), v))
                  for v in itertools.product(*param_grid.values())]
    print(f"{len(all_params)} hyperparameter combinations")

    # Setup for logging
    console_output_filename = f'./output/{embedding_param}_{task_param}_console_output.txt'
    with open(console_output_filename, 'a') as logfile:
        logfile.write('\n\nBEGIN TRAINING LOOP\n\n')

    # Setup for saving results
    results_folder = pathlib.Path(f"results/{embedding_param}/{task_param}")
    results_folder.mkdir(parents=True, exist_ok=True)
    save_file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_folder / f"val_{save_file_id}_{model_param}.csv"

    if task_config.class_type in ["BC", "MC"]:
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = list(all_params[0].keys())
            writer.writerow(
                ['mcc', 'f1', 'accuracy', 'training time', 'training energy'] + headers)
        print(f"saving results to ./{results_file}")
        # Saves best accuracy for progress bar display
        best_acc = 0.0
    elif task_config.class_type == "R":
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = list(all_params[0].keys())
            writer.writerow(
                ['pearson', 'spearman', 'training time', 'training energy'] + headers)
        print(f"saving results to ./{results_file}")
        # Saves best accuracy for progress bar display
        best_pearson = -2.0

    # Iterate over all combinations of hyperparameters
    bar = tqdm(enumerate(all_params), total=len(all_params))
    for i, params in bar:
        # Formatting params to display
        print_params = params.copy()
        for param in ['category', 'device']:
            del print_params[param]

        # Initialize the model with current set of hyperparameters
        feed_forward = FeedForward(**params)

        metrics, train_times_per_epoch, energy_per_epoch = feed_forward.fit(
            X_train, Y_train, X_val, Y_val)
        training_time = np.mean(train_times_per_epoch)
        training_energy = np.mean(energy_per_epoch)
        # Log average training time per epoch for current parameter set
        # Note: FFN frequently stops early

        if task_config.class_type in ["BC", "MC"]:
            epoch, val_loss, val_accuracy, val_f1, val_mcc = metrics["epoch"], metrics[
                "loss"], metrics["acc"], metrics["f1"], metrics["mcc"]
            best_acc = max(best_acc, val_accuracy)
            bar.set_description(
                f"Best Acc: {best_acc:.5f}, Last test: {val_accuracy:.5f}")

            # Write stats to log file
            with open(console_output_filename, 'a') as logfile:
                logfile.write(f"\n\nTraining with parameters:\n{print_params}")
                logfile.write(f"\nEarly stopped on epoch: {epoch}")
                logfile.write(f"\nValidation accuracy: {val_accuracy}")
                logfile.write(f"\nValidation f1-score: {val_f1}")
                logfile.write(f"\nValidation MCC     : {val_mcc}")
                logfile.write(f"\nTraining time      : {training_time}")
                logfile.write(f"\nTraining energy    : {training_energy}")
            # Write to results csv
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([val_mcc, val_f1, val_accuracy,
                                training_time, training_energy] + list(params.values()))

        elif task_config.class_type == "R":  # just report loss for regression task
            epoch, val_loss, val_pearson, val_spearman = metrics[
                "epoch"], metrics["loss"], metrics["pearson"], metrics["spearman"]
            best_pearson = max(best_pearson, val_pearson)
            bar.set_description(
                f"Best Pearson's corrcoef: {best_pearson:.3f}, Last test: {val_pearson:.3f}")

            # Write stats to log file
            with open(console_output_filename, 'a') as logfile:
                logfile.write(f"\n\nTraining with parameters:\n{print_params}")
                logfile.write(f"\nEarly stopped on epoch: {epoch}")
                logfile.write(f"\nValidation loss: {val_loss}")
                logfile.write(
                    f"\nValidation Pearson's corrcoef: {val_pearson}")
                logfile.write(
                    f"\nValidation Spearman's corrcoef: {val_spearman}")
                logfile.write(f"\nTraining time      : {training_time}")
                logfile.write(f"\nTraining energy    : {training_energy}")
            # Write to results csv
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([val_pearson, val_spearman, training_time,
                                training_energy] + list(params.values()))
