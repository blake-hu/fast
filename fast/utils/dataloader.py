from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


# Function to merge datasets
def clean_dataset(inputs, split):
    common_columns_to_remove = {'sentence', 'sentence1', 'sentence2', 'label', 'question1', 'question2', 'premise', 'hypothesis', 'question'}

    if isinstance(inputs, list):
        dataframe = [ds[split].to_pandas() for ds in inputs]
        dataframe = [df.drop(columns=common_columns_to_remove, errors='ignore') for df in dataframe]
        cleaned_df = dataframe[0]
        for df in dataframe[1:]:
            cleaned_df = pd.merge(cleaned_df, df, on='idx', how='inner')
    else:
        dataframe = inputs[split].to_pandas()
        cleaned_df = dataframe.drop(columns=common_columns_to_remove, errors='ignore')

    return cleaned_df

class TransformerDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Exclude 'idx' column and convert others to tensors
        self.columns = [col for col in dataframe.columns if col != 'idx']
        for col in self.columns:
            # Convert to NumPy array first for efficiency
            np_array = np.array(dataframe[col].tolist())
            setattr(self, col, torch.tensor(np_array, dtype=torch.int64))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve tensors by index for each column
        item = {col: getattr(self, col)[idx] for col in self.columns}
        return item
    
