import llama
import torch

DEVICE = "cuda"

MAX_SEQ_LEN: int = 2048
BATCH_SIZE: int = 16
VALID_BATCH_SIZE: int = 1
EPOCHS = 100

MODEL_DIM = 256
MODEL_N_HEADS = 8
MODEL_N_LAYERS = 8

TOKENIZER_PATH="/mmfs1/gscratch/scrubbed/arprieve/llama_data/tokenizer.model"
MODEL_PATH="/gscratch/scrubbed/ebdaniel/llama/models/baseline/model_epoch_14.pt"

tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)

model_args: llama.ModelArgs = llama.ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=BATCH_SIZE,
    dim=MODEL_DIM,
    n_heads=MODEL_N_HEADS,
    n_layers=MODEL_N_LAYERS,
    device=DEVICE)
model = llama.Transformer(model_args)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Loaded model")

if torch.cuda.is_available():
    model = model.cuda()

generator = llama.LLaMA(model, tokenizer)

print("Loaded generator")

empty_prompt_str = tokenizer.decode([tokenizer.bos_id])

prompts = [empty_prompt_str, "def bar(x):", "def baz(x):"]

generated = generator.generate(prompts, max_gen_len=100)

print("Generated")

for prompt, gen in zip(prompts, generated):
    print("Prompt:", prompt)
    print("Generated:", gen)
    print()

