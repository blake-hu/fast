"""
Trains a nn.TransformerEncoder model, which assigns a probability for the likelihood of a given word to follow a sequence of words
nn.TrasnformerEncoder consists of multiple layers of nn.TransformerEncoderLayer
Use square mask attention to attend only to earlier positions in sequence
Then passed through linear layer to output unnormalized logits
"""
import math
import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from typing import Tuple, Optional
from copy import deepcopy
from tempfile import TemporaryDirectory
import os
import matplotlib.pyplot as plt



from util import *

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import time


"""
nn.TransformerEncoderLayer is self attention and feedforward network
nn.Embedding is a lookup table that stores embeddings of a fixed dictionary and size. Stores word embeddings, retrieve them with indices
ntoken: number of embeddings, aka size of vocabulary
d_model: embedding dimension, aka number of expected features in the input
nhead: number of heads for multiheadattention
d_hid: dimension for feedforward network
nlayers: number of transformer encoder layers
dropout: dropout for positional encoder and feedforward
"""

class EncoderModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)     # positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)   # hyperparams for each encoder layer
        # self.classifier = 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)     # encoder using encoder layers
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken, bias=True)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)     # multiply to prevent embeddings from becoming too small
        src = self.pos_encoder(src)                             # add positional encoding to input embeddings, uses dropout
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output




train_iter = WikiText2(split='train')               # returns an iterable over a list of strings
tokenizer = get_tokenizer('basic_english')          # function that tokenizes a string into a list of words; basic_english normalizes the string first and splits by space
# vocab maps tokens to indices
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials = ['<unk>']) # build vocab from iterator, returns torchtext.vocab.Vocab object
vocab.set_default_index(vocab['<unk>'])             # default index returned when OOV token queried



train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter, vocab, tokenizer)
val_data = data_process(val_iter, vocab, tokenizer)
test_data = data_process(test_iter, vocab, tokenizer)

device = torch.device('cpu')


batch_size = 20     # num batches
eval_batch_size = 10    # num batches
train_data = batchify(train_data, batch_size).to(device)   # return shape: length of batch x num batches
val_data = batchify(val_data, eval_batch_size).to(device)
test_data = batchify(test_data, eval_batch_size).to(device)


bptt = 35       # size of data chunk for a batch
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in "nn.TransformerEncoder"
nlayers = 2  # number of ""nn.TransformerEncoderLayer" in "nn.TransformerEncoder"
nhead = 2  # number of heads in "nn.MultiheadAttention"
dropout = 0.2  # dropout probability
# model = HopfieldTransformer(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model = EncoderModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


criterion = nn.CrossEntropyLoss()
lr = 5.0    # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model : nn.Module):
    model.train()   # turns on train mode
    epoch_losses = []
    epoch_accuracies = []
    total_loss = 0
    log_interval = 200
    num_batches = len(train_data) // bptt
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)      # 35x20, 700
        output = model(data)
        output_flat = output.view(-1, ntokens)      # makes it 700xlen(vocab)
        loss = criterion(output_flat, targets)
        
        if batch % 10 == 0:
            acc = accuracy(output_flat, targets)
            epoch_accuracies.append(acc)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clips gradients to prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            epoch_losses.append(cur_loss)
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | accuracy {acc} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    return epoch_losses, epoch_accuracies

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


losses = []
accuracies = []
best_val_loss = float('inf')
epochs = 1

# Use Temporary Directory to save the best model parameters
with TemporaryDirectory() as tempdir:
    best_model_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        l, a = train(model)
        losses.extend(l)
        accuracies.extend(a)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    model.load_state_dict(torch.load(best_model_path))

plt.figure(1)
plt.plot(torch.arange(len(losses)), losses)
plt.title("loss per 200 training iterations")
plt.figure(2)
plt.plot(torch.arange(len(accuracies)), accuracies)
plt.title("accuracy per iteration")
plt.show()

# add accuracy graph