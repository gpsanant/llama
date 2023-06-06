import llama

TOKENIZER_PATH="llama/tokenizer"
MODEL_PATH="llama/model"


tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)
model = llama.Transformer(model_path=MODEL_PATH)

generator = llama.LLaMA(model, tokenizer)

empty_prompt_str = tokenizer.decode([tokenizer.bos_id])

prompts = [empty_prompt_str, "def bar(x):", "def baz(x):"]

generated = generator.generate(prompts, max_gen_len=100)


