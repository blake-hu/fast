import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.utils.data import dataset
from typing import Tuple

"""
positional encodings have the same dimension as the embeddings so that the two can be summed. What does sin and cos terms do?
d_model:
dropout:
max_len: max number of vectors needed to be positionally encoded at a time
"""

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # registers a buffer, which does not appear in module.parameters and does not update at every step, like parameters

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """converts raw text into a flat tensor"""
    data = [torch.tensor(vocab(tokenizer(item)), dtype = torch.long) for item in raw_text_iter]
    filtered = filter(lambda t : t.numel() > 0, data)   # returns an iterator for a filtered list of list of tokenized words
    return torch.cat(tuple(filtered))   # cat concatenates the sequence of tensors into one long sequence



def batchify(data : Tensor, bsz: int) ->Tensor:
    "Divides data into bsz separate sequences, removes extra elements that wouldn't cleanly fit"
    seq_len = data.size(0) // bsz
    data = data[:seq_len*bsz]
    data = data.view(bsz, seq_len).t()  # reshape data so the token in each column is succeeded by the token beneath it
    data = data.contiguous()     # contiguous makes a copy of the tensor such that the order of its elements is the same as in memory
    return data

def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    # get input data and target data. Target data are the batches right after input data, in shape [seq_len * batch_size]
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)            # reshape to vector to row vector
    return data, target


def accuracy(source: Tensor, targets: Tensor):
    soft = torch.nn.functional.softmax(source, dim = 1)
    pred = torch.argmax(soft, dim = 1)  # turns matrix to vector
    matches = torch.sum(pred==targets).item()
    return matches/torch.numel(targets)
