import argparse
import llama
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--little", action="store_true", help="Activate the --little flag.")
parser.add_argument("--tokenizer-path", type=str, help="Provide a path to the tokenizer.")
parser.add_argument("--model-path", type=str, help="Provide a path to the model.")
parser.add_argument("--temperature", type=float, help="Provide a temperature for generation.")
parser.add_argument("--max-gen-len", type=int, help="Provide a maximum generation length.")
parser.add_argument("--prompt", type=str, help="Provide a prompt for generation.")
args = parser.parse_args()

DEVICE = "cuda"

MAX_SEQ_LEN: int = 2048
BATCH_SIZE: int = 16
VALID_BATCH_SIZE: int = 1
EPOCHS = 100

MODEL_DIM = 256
MODEL_N_HEADS = 8
MODEL_N_LAYERS = 8

# we default to paths on hyak

TOKENIZER_PATH="/mmfs1/gscratch/scrubbed/arprieve/llama_data/tokenizer.model"
if args.tokenizer_path:
    TOKENIZER_PATH = args.tokenizer_path

MODEL_PATH="/gscratch/scrubbed/ebdaniel/llama/models/baseline/model_epoch_14.pt"
if args.model_path:
    MODEL_PATH = args.model_path
elif args.little:
    MODEL_PATH="/gscratch/scrubbed/ebdaniel/llama/models/little/model_epoch_14.pt"

tokenizer = llama.Tokenizer(model_path=TOKENIZER_PATH)

if args.little:
    MODEL_DIM = 128

model_args: llama.ModelArgs = llama.ModelArgs(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=BATCH_SIZE,
    dim=MODEL_DIM,
    n_heads=MODEL_N_HEADS,
    n_layers=MODEL_N_LAYERS,
    device=DEVICE)
model_args.vocab_size = tokenizer.n_words

model = llama.TransformerInference(model_args)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Loaded model")

if torch.cuda.is_available():
    model = model.cuda()

generator = llama.LLaMA(model, tokenizer)

print("Loaded generator")

empty_prompt_str = tokenizer.decode([tokenizer.bos_id])

prompts = [
    empty_prompt_str,
    "A \"Burringo\" is a car with very fast acceleration. An example of a sentence that uses the word Burringo is:",
    "Poor English input: The patient was died.",
    "Plenty of people think that "
]
if args.prompt:
    prompts.append(args.prompt)


temperature = 0.8
if args.temperature:
    temperature = args.temperature

max_gen_len = 100
if args.max_gen_len:
    max_gen_len = args.max_gen_len

generated = generator.generate(prompts, max_gen_len, temperature=temperature)

print("Generated")

for prompt, gen in zip(prompts, generated):
    print("Prompt:", prompt)
    print("Generated:", gen)
    print()