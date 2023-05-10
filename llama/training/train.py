from llama.llama.tokenizer import Tokenizer
from llama.llama.training.loader import load_data
from llama.llama.training.optimizer import OptimizerArgs, create_optimizer_and_scheduler
import torch
from llama.model import Transformer, ModelArgs
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming the model class is already imported:
# from your_module import Transformer

tokenizer = Tokenizer(model_path='path/to/tokenizer.model')

model_args = ModelArgs()
model = Transformer(model_args)
model = model.to('cuda') # Assuming you're using GPU

criterion = nn.CrossEntropyLoss()
optimizer_args = OptimizerArgs(learning_rate=0.001, weight_decay=0.1, warmup_steps=2000, total_steps=10000)
optimizer, scheduler = create_optimizer_and_scheduler(model, optimizer_args)

# Assuming your dataloader is already defined
dataloader = DataLoader(load_data("../data/val.jsonl", tokenizer), batch_size=32, shuffle=True)

for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(tqdm(dataloader), 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to('cuda'), data[1].to('cuda') 

        # Forward pass
        outputs = model(inputs, start_pos=0)
            
        # Compute loss
        loss = criterion(outputs, labels)
            
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Avg. Loss: {running_loss / 200}')
            running_loss = 0.0

print('Finished Training')