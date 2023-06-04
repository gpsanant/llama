# Copyright (c) Aric Prieve Inc. and totally all of my affiliates.

# This is the training for CSE 493s HW2
# I run this file using: `torchrun llama/train.py` from the llama home folder
# Be sure to set paths below to the data and tokenizer model
# (can be downloaded from ed)

import os
import sys
import pandas as pd
import time
import torch
import llama
import math
import csv

from typing import Tuple
from torch.utils.data import dataset
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

DEVICE = "cpu"
TOKENIZER_PATH = "tokenizer.model"
TRAIN_DATA_PATH = "../llama_data/test.jsonl.zst"
NUM_TRAIN_DATA = 40000
VALID_DATA_PATH = "../llama_data/test.jsonl.zst"
NUM_VALID_DATA = 10000

MAX_SEQ_LEN: int = 2048
BATCH_SIZE: int = 32

MODEL_DIM = 512
MODEL_N_HEADS = 4
MODEL_N_LAYERS = 4


#######################################################
### PREPARE DATASETS ##################################
#######################################################

def data_process(raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
    # Tokenize and sort by number of tokens in sequence
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in raw_text_iter]
    prompt_tokens.sort(key=len)
  
    max_prompt_size = max([len(t) for t in prompt_tokens])

    tokens = torch.full((len(prompt_tokens), max_prompt_size), tokenizer.pad_id).to(DEVICE).long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()        
    return tokens

def get_batch(source: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:    
    idx = i * BATCH_SIZE
    
    # Get the sequence capped at a certain length
    data = source[idx:idx+BATCH_SIZE, :MAX_SEQ_LEN]
    
    # Figure out minimum lengths to truncate to avoid -1 values
    min_lens = torch.argmin(data, dim=1)
    min_lens = min_lens[min_lens > 0]
    
    seq_len = MAX_SEQ_LEN
    if (len(min_lens) > 0):
        seq_len = torch.min(min_lens) - 1
        
    # Add 1 to fix -1 values to 0 so one_hot doesn't get mad
    target = source[idx:idx+BATCH_SIZE:, 1:seq_len+1] + 1
    targets = torch.nn.functional.one_hot(target, num_classes=tokenizer.n_words + 1)
    
    # Return to correct tokens by removing the first row
    return data[:,:seq_len], targets[:,:,1:].to(DEVICE).type(torch.FloatTensor)

start_time = time.time()
print("Loading Datasets")

# Read in test data from Pile (this is just a start)
# Download this according to the project instructions
train_df = pd.read_json(TRAIN_DATA_PATH, lines=True, nrows=NUM_TRAIN_DATA)
valid_df = pd.read_json(VALID_DATA_PATH, lines=True, nrows=NUM_VALID_DATA)

tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)

train_data = data_process(train_df['text'])
valid_data = data_process(valid_df['text'])

print(f"Loaded in {time.time() - start_time:.2f} seconds")


#######################################################
### PREPARE MODEL #####################################
#######################################################

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo", rank=local_rank, world_size=world_size) # TODO: or nccl for gpu
    initialize_model_parallel(world_size)
    # torch.cuda.set_device(local_rank) # TODO: I don't have a gpu on my laptop

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

# Start time similar to example.py
start_time = time.time()
print("Loading Model")

local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

model_args: llama.ModelArgs = llama.ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=BATCH_SIZE,
    dim=MODEL_DIM,
    n_heads=MODEL_N_HEADS,
    n_layers=MODEL_N_LAYERS)
model_args.vocab_size = tokenizer.n_words

# torch.set_default_tensor_type(torch.cuda.HalfTensor) # TODO: I don't have a gpu on my laptop
model = llama.Transformer(model_args)
torch.set_default_tensor_type(torch.FloatTensor)

print(f"Loaded in {time.time() - start_time:.2f} seconds")


#######################################################
### TRAIN MODEL #######################################
#######################################################

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

train_losses = []
valid_losses = [float('inf')]

# Randomize model weights
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

def train(model: torch.nn.Module) -> None:
    total_loss = 0.
    start_time = time.time()
    num_train_batches = len(train_data) // BATCH_SIZE
    num_valid_batches = len(valid_data) // BATCH_SIZE
    log_interval = 125
    epochs = 10
    
    for epoch in range(epochs):
        model.train()  # turn on train mode
        
        for batch in range(num_train_batches):
            data, targets = get_batch(train_data, batch)
            output = model(data, 0)
            
            loss = criterion(output.view(-1, tokenizer.n_words),
                             targets.view(-1, tokenizer.n_words))
            loss.requires_grad = True
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_train_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.5f}')
                total_loss = 0
                start_time = time.time()
                
                train_losses.append(cur_loss)
                
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for batch in range(num_valid_batches):
                data, targets = get_batch(valid_data, batch)
                output = model(data, 0)
                
                loss = criterion(output.view(-1, tokenizer.n_words),
                                 targets.view(-1, tokenizer.n_words))
                total_loss += loss.item()
                
        if min(valid_losses) < total_loss / NUM_VALID_DATA:
            torch.save(model.state_dict(), "my_model")
                
        valid_losses.append(total_loss / NUM_VALID_DATA)

print("Starting to train model!")
start_time = time.time()
train(model)
print(f"Trained in {time.time() - start_time:.2f} seconds")

print(train_losses)
print(valid_losses)

# Write out results to a csv for plotting later
file = open('train_losses.csv', 'w+', newline ='')
with file:   
    write = csv.writer(file)
    write.writerow(train_losses)
    
file = open('valid_losses.csv', 'w+', newline ='')
with file:   
    write = csv.writer(file)
    write.writerow(valid_losses[1:]) 

