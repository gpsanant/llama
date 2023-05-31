# Copyright (c) Aric Prieve Inc. and totally all of my affiliates.

# This is the training for CSE 493s HW2
# I run this file using: `torchrun llama/train.py` from the llama home folder
# Be sure to set paths below to the data and tokenizer model
# (can be downloaded from ed)

import os
import sys
import json
import pandas as pd
import time
import torch
import llama
import math

from typing import Tuple
from torch.utils.data import dataset
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

DATA_PATH = "../llama_data/test.jsonl.zst"
TOKENIZER_PATH = "tokenizer.model"
DEVICE = "cpu"
NUMBER_DATA_POINTS = 2000

MAX_SEQ_LEN: int = 100 # TODO: DECIDE THIS VALUE, IF AT ALL, maybe 512?
BATCH_SIZE: int = 4

#######################################################
### PREPARE DATASETS ##################################
#######################################################

def data_process(raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in raw_text_iter]
    
    max_prompt_size = max([len(t) for t in prompt_tokens])
    total_len = max_prompt_size # TODO: decide this value, min(max_seq_len, max_prompt_size)

    tokens = torch.full((len(prompt_tokens), total_len), tokenizer.pad_id).to(DEVICE).long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    return tokens

def get_batch(source: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = MAX_SEQ_LEN # TODO: decide this based on seq len, could be: min(max_seq_len, len(source) - 1 - i)
    idx = i * BATCH_SIZE
    
    # Get the sequence capped at a certain length
    data = source[idx:idx+BATCH_SIZE, :seq_len]
    
    # Add 1 to fix -1 values to 0 so one_hot doesn't get mad
    target = source[idx:idx+BATCH_SIZE:, 1:seq_len+1] + 1
    targets = torch.nn.functional.one_hot(target, num_classes=tokenizer.n_words + 1)
    
    # Return to correct tokens by removing the first row
    return data, targets[:,:,1:].to(DEVICE).type(torch.FloatTensor)

start_time = time.time()
print("Loading Datasets")

# Read in test data from Pile (this is just a start)
# Download this according to the project instructions
json = pd.read_json(DATA_PATH, lines=True, nrows=NUMBER_DATA_POINTS)

tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)
train_iter = json['text']

train_data = data_process(train_iter)

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
    max_batch_size=BATCH_SIZE)
model_args.vocab_size = tokenizer.n_words

# torch.set_default_tensor_type(torch.cuda.HalfTensor) # TODO: I don't have a gpu on my laptop
model = llama.Transformer(model_args)
torch.set_default_tensor_type(torch.FloatTensor)

print(f"Loaded in {time.time() - start_time:.2f} seconds")


#######################################################
### TRAIN MODEL #######################################
#######################################################

criterion = torch.nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: torch.nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    num_batches = len(train_data) // BATCH_SIZE
    
    for batch in range(num_batches):
        data, targets = get_batch(train_data, batch)

        output = model(data, 0)
        loss = criterion(output, targets) # TODO: Should i flatten?
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
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

train(model)



