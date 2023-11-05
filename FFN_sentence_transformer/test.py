import os
import numpy as np
import warnings
import itertools

from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_dataset
from transformers import BertTokenizer
import torch
from utils.adapter import BERTAdapter
from utils.feed_forward import FeedForward
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Device ##########################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
#######################################################################

# Load Data
data = load_dataset("glue", "cola")

# Initialize the tokenizer from the BERT base uncased model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
mpnetv2 = SentenceTransformer("all-mpnet-base-v2").to(device)

# Define a function to tokenize a batch of texts


# def encode_batch(batch):
#     """Tokenizes a batch of texts using the pre-initialized tokenizer."""
#     return tokenizer(batch["sentence"], truncation=True, padding="max_length")


X_train = mpnetv2.encode(data["train"]["sentence"])
X_val = mpnetv2.encode(data["validation"]["sentence"])
# X_test = mpnetv2.encode(data["test"]["sentence"])

Y_train = np.array(data["train"]["label"])
Y_val = np.array(data["validation"]["label"])
# Y_test = np.array(data["test"]["label"])

param_grid = {
    'num_epochs': [100],
    'batch_size': [32, 128, 512],
    'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
    'category': ['C'],
    'norm': [False],
    'size': [768],
    'num_layers': [1, 3, 5, 10],
    'weight_decay': [1e-2, 1e-3, 1e-4, 1e-5],
    'patience': [3],
    'min_delta': [0],
    'device': [device]
}

# Create a list of all combinations of hyperparameters
all_params = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

best_params = None
highest_val_accuracy = 0

# Iterate over all combinations of hyperparameters
for params in all_params:
    print("Training with parameters:", params)
    # Initialize the model with current set of hyperparameters
    feed_forward = FeedForward(**params)

    _, _, val_accuracy = feed_forward.fit(X_train, Y_train, X_val, Y_val)
    print("Validation accuracy:", val_accuracy)

    # Save the parameters if they provide a better accuracy
    if val_accuracy > highest_val_accuracy:
        highest_val_accuracy = val_accuracy
        best_params = params

# Print the best parameters
print("Best Parameters:", best_params)
print("Highest Validation Accuracy:", highest_val_accuracy)
