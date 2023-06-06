import llama
import torch

TOKENIZER_PATH="/mmfs1/gscratch/scrubbed/arprieve/llama_data/tokenizer.model"
MODEL_PATH="/gscratch/scrubbed/ebdaniel/llama/models/baseline/model_epoch_14.pt"

tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)
model = llama.Transformer(model_path=MODEL_PATH)

if torch.cuda.is_available():
    model = model.cuda()

generator = llama.LLaMA(model, tokenizer)

empty_prompt_str = tokenizer.decode([tokenizer.bos_id])

prompts = [empty_prompt_str, "def bar(x):", "def baz(x):"]

generated = generator.generate(prompts, max_gen_len=100)


