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

DEVICE = "cuda"
TOKENIZER_PATH = "/mmfs1/gscratch/scrubbed/arprieve/llama_data/tokenizer.model"
TRAIN_DATA_PATH = "/mmfs1/gscratch/scrubbed/arprieve/llama_data/00.jsonl.zst"
NUM_TRAIN_DATA = 20000
VALID_DATA_PATH = "/mmfs1/gscratch/scrubbed/arprieve/llama_data/val.jsonl.zst"
NUM_VALID_DATA = 10000

MAX_SEQ_LEN: int = 2048
BATCH_SIZE: int = 16
VALID_BATCH_SIZE: int = 1
EPOCHS = 100

MODEL_DIM = 256
MODEL_N_HEADS = 8
MODEL_N_LAYERS = 8

OUTPUT_DIR = r"/mmfs1/gscratch/scrubbed/ebdaniel/llama/models"
MODEL_NAME = "standard"

# Make sure everything is divisible by batch size
NUM_TRAIN_DATA = NUM_TRAIN_DATA // BATCH_SIZE * BATCH_SIZE
# NUM_VALID_DATA = NUM_VALID_DATA // BATCH_SIZE * BATCH_SIZE

#######################################################
### PREPARE DATASETS ##################################
#######################################################

def data_process(raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
    # Tokenize and sort by number of tokens in sequence
    print("starting tokenizing", time.time() - start_time)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in raw_text_iter]
    print("finished tokenizing, start truncating", time.time() - start_time)
    for prompt in prompt_tokens:
        del prompt[MAX_SEQ_LEN + 1:]
    print("finished truncating, start sorting", time.time() - start_time)
    prompt_tokens.sort(key=len)
    print("finish sorting", time.time() - start_time)

    tokens = torch.full((len(prompt_tokens), MAX_SEQ_LEN + 1), tokenizer.pad_id).cpu().long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    return tokens.cpu()

def get_batch(source: torch.Tensor, i: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = i * batch_size

    # Get the sequence capped at a certain length
    data = source[idx:idx+batch_size, :MAX_SEQ_LEN]

    # Figure out minimum lengths to truncate to avoid -1 values
    min_lens = torch.argmin(data, dim=1)
    min_lens = min_lens[min_lens > 0]

    seq_len = MAX_SEQ_LEN
    if (len(min_lens) > 0):
        seq_len = torch.min(min_lens) - 1

    # Add 1 to fix -1 values to 0 so one_hot doesn't get mad
    target = source[idx:idx+batch_size, 1:seq_len+1]
    target[target < 0] = 0
    targets = torch.nn.functional.one_hot(target, num_classes=tokenizer.n_words)

    # Return to correct tokens by removing the first row
    return data[:,:seq_len].to(DEVICE), targets.type(torch.cuda.FloatTensor).to(DEVICE)

start_time = time.time()
print("Loading Datasets", start_time)

# Read in test data from Pile (this is just a start)
# Download this according to the project instructions
train_df = pd.read_json(TRAIN_DATA_PATH, lines=True, nrows=NUM_TRAIN_DATA)
valid_df = pd.read_json(VALID_DATA_PATH, lines=True, nrows=NUM_VALID_DATA)
print("Finished reading json", time.time() - start_time)
valid_df = valid_df.loc[valid_df['text'].str.len() > 10]
train_df = train_df.loc[train_df['text'].str.len() > 10]

tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)

train_data = data_process(train_df['text']).cpu()
valid_data = data_process(valid_df['text']).cpu()

print("cuda device: ", torch.cuda.get_device_name(0))
print("train_data device", train_data.get_device())
print("valid_data device", valid_data.get_device())
print(f"Loaded in {time.time() - start_time:.2f} seconds")


#######################################################
### PREPARE MODEL #####################################
#######################################################

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl", rank=local_rank, world_size=world_size) # TODO: or nccl for gpu
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank) # TODO: I don't have a gpu on my laptop

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
    n_layers=MODEL_N_LAYERS,
    device=DEVICE)
model_args.vocab_size = tokenizer.n_words

# torch.set_default_tensor_type(torch.cuda.HalfTensor) # TODO: I don't have a gpu on my laptop
model = llama.Transformer(model_args)
torch.set_default_tensor_type(torch.FloatTensor)

print(f"Loaded in {time.time() - start_time:.2f} seconds")


#######################################################
### TRAIN MODEL #######################################
#######################################################

lr = 0.0001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-9)

sum_train_losses = []
avg_train_losses = []
train_times = []
# set so model will always be saved after first epoch
sum_valid_losses = []
avg_valid_losses = []

# Randomize model weights
#for p in model.parameters():
#    if p.dim() > 1:
#        torch.nn.init.xavier_uniform_(p)

def train(model: torch.nn.Module) -> None:
    total_loss = 0.
    start_time = time.time()
    num_train_batches = len(train_data) // BATCH_SIZE
    # num_valid_batches = len(valid_data) // BATCH_SIZE
    # num valid batches is 1, we iterate through each datapoint
    log_interval = 25

    for epoch in range(EPOCHS):
        model.train()  # turn on train mode
        epoch_train_loss = 0.
        epoch_valid_loss = 0.

        epoch_train_start_time = time.time()

        for batch in range(num_train_batches):
            data, targets = get_batch(train_data, 63, BATCH_SIZE)
            if  (data.shape[1] < 10):
                del data
                del targets
                torch.cuda.empty_cache()
                continue
            optimizer.zero_grad()
            output = model(data, 0)
            loss = criterion(output.view(-1, tokenizer.n_words),
                             targets.view(-1, tokenizer.n_words))
            # loss.requires_grad = True

            loss.backward()
            # print("loss", loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            total_loss += loss.item()
            epoch_train_loss += loss.item()

            del data
            del targets
            del loss
            torch.cuda.empty_cache()

            if batch % log_interval == 0 and batch > 0:
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_train_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.5f}')
                total_loss = 0
                start_time = time.time()

        total_loss = 0.
        sum_train_losses.append(epoch_train_loss)
        avg_train_losses.append(epoch_train_loss / num_train_batches)
        print(f'| epoch {epoch:3d} | '
            f'summed training loss for epoch {epoch_train_loss:5.5f} | time elapsed: {time.time() - epoch_train_start_time:.2f} seconds')
        train_times.append(time.time() - epoch_train_start_time)
        print(f'| epoch {epoch:3d} | '
            f'average per batch training loss for epoch {epoch_train_loss / num_train_batches:5.5f}')

        model.eval()  # turn on evaluation mode
        with torch.no_grad():
            for batch in range(len(valid_data)):
                data, targets = get_batch(valid_data, batch, 1)
                if (data.shape[1] < 1):
                    del data
                    del targets
                    torch.cuda.empty_cache()
                    continue
                output = model(data, 0)

                loss = criterion(output.view(-1, tokenizer.n_words),
                                 targets.view(-1, tokenizer.n_words))
                epoch_valid_loss += loss.item()
                del data
                del targets
                del loss
                torch.cuda.empty_cache()
        print(f'| epoch {epoch:3d} | '
            f'summed validation loss over epoch {epoch_valid_loss:5.5f}')
        print(f'| epoch {epoch:3d} | '
            f'averaged per batch validation loss over epoch {epoch_valid_loss / len(valid_data):5.5f}')

        torch.save(model.state_dict(), model_dir + "model_epoch_" + str(epoch) + '.pt')

        sum_valid_losses.append(epoch_valid_loss)
        avg_valid_losses.append(epoch_valid_loss / len(valid_data))

        write_progress()

# a function for writing our to all of the csvs
def write_progress():
    file = open(model_dir + 'sum_train_losses.csv', 'w', newline ='')
    with file:
        write = csv.writer(file)
        write.writerow(sum_train_losses)

    file = open(model_dir + 'average_per_batch_train_losses.csv', 'w', newline ='')
    with file:
        write = csv.writer(file)
        write.writerow(avg_train_losses)

    file = open(model_dir + 'train_times.csv', 'w', newline ='')
    with file:
        write = csv.writer(file)
        write.writerow(train_times)

    file = open(model_dir + 'sum_valid_losses.csv', 'w', newline ='')
    with file:
        write = csv.writer(file)
        write.writerow(sum_valid_losses)

    file = open(model_dir + 'average_per_batch_valid_losses.csv', 'w', newline ='')
    with file:
        write = csv.writer(file)
        write.writerow(avg_valid_losses)


# Create model directory to write in
model_dir = OUTPUT_DIR + '/' + MODEL_NAME + '/'
os.makedirs(model_dir)

print("Starting to train model!")

print("BATCH_SIZE: " + str(BATCH_SIZE) + "\n")
print("EPOCHS: " + str(EPOCHS) + "\n")
print("MODEL_DIM: " + str(MODEL_DIM) + "\n")
print("NUM_HEADS: " + str(MODEL_N_HEADS) + "\n")
print("NUM_LAYERS: " + str(MODEL_N_LAYERS) + "\n")
print("NUM_TRAIN_DATA: " + str(len(train_data)) + "\n")
print("NUM_VALID_DATA: " + str(len(valid_data)) + "\n")
# create a text file with all hyperparameters in it
file = open(model_dir + 'hyperparameters.txt', 'w+')
file.write("BATCH_SIZE: " + str(BATCH_SIZE) + "\n")
file.write("EPOCHS: " + str(EPOCHS) + "\n")
file.write("MODEL_DIM: " + str(MODEL_DIM) + "\n")
file.write("NUM_HEADS: " + str(MODEL_N_HEADS) + "\n")
file.write("NUM_LAYERS: " + str(MODEL_N_LAYERS) + "\n")
file.write("NUM_TRAIN_DATA: " + str(len(train_data)) + "\n")
file.write("NUM_VALID_DATA: " + str(len(valid_data)) + "\n")

total_start_time = time.time()
train(model.to(DEVICE))
print(f"Trained in {time.time() - total_start_time:.2f} seconds")

# Write out results to a csv for plotting later
write_progress()
