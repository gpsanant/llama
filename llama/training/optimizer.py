from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

class OptimizerArgs:
    def __init__(self, learning_rate, weight_decay, warmup_steps, total_steps):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

def create_optimizer_and_scheduler(model, optimizer_args):
    # Create the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=optimizer_args.learning_rate, betas=(0.9, 0.95), weight_decay=optimizer_args.weight_decay)
    
    # Create the learning rate scheduler.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=optimizer_args.warmup_steps, 
        num_training_steps=optimizer_args.total_steps, 
        last_epoch=-1, 
        num_cycles=0.5
    )
    
    return optimizer, scheduler
