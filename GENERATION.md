# LLaMA Generation Script

This script uses the LLaMA model to generate text based on the provided prompts.

## NOTE: Inference only works on branch `inference`.

## Requirements

- Python 3.7 or higher
- PyTorch
- CUDA-enabled GPU (optional)

## Usage

You can run this script from the command line using the following syntax:

```sh
python llama/inference.py [arguments]
```


### Arguments

- `--little`: This flag adjusts the dimensions of the model if provided. Default: baseline.

- `--tokenizer-path`: Provide the path to the tokenizer model. Default: Works on hyak.

- `--model-path`: Provide the path to the transformer model. Default: Works on hyak.

- `--temperature`: A float value that defines the temperature for generation. Higher values (e.g., 1.0 or higher) lead to more diverse outputs. Lower values (e.g., 0.1) make the outputs more deterministic and less diverse. Default: 0.8.

- `--max-gen-len`: An integer value that specifies the maximum length of the generated text. Default: 100.

- `--prompt`: Provide a prompt for text generation.

For example:

```shell
python inference.py --tokenizer-path /mmfs1/gscratch/scrubbed/arprieve/llama_data/tokenizer.model --model-path ../models/baseline/model_epoch_14.pt --temperature 1 --max-gen-len 75 --prompt "What should I make my xbox gamertag"
```

or, for the little model,

```shell
python inference.py --little --tokenizer-path /mmfs1/gscratch/scrubbed/arprieve/llama_data/tokenizer.model --
model-path ../models/little/model_epoch_14.pt --temperature 1 --max-gen-len 75 --prompt "What should I make my xbox gamertag"
```