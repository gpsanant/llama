# LLaMA 

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

### Additional setup to replicate HW2 experiments
```
pip install zstandard
```
In order to run the different experiments, you can either change the model parameters, which are the global variables at the top of train.py, or you can run the two training scripts with parameters we have set up. train.py corresponds to our baseline, and train_little.py corresponds to our little graph.

At the top of these training files, modifying the following variables with your desired values:
```python
TOKENIZER_PATH = # path to tokenizer model
TRAIN_DATA_PATH = # path to training data file
NUM_TRAIN_DATA = # number of training data points to train on per epoch
VALID_DATA_PATH = # path to validation data file
NUM_VALID_DATA = # number of validation data points to evaluate on per epoch

MAX_SEQ_LEN: int = # the maximum input size (in tokens) the model will accept
BATCH_SIZE: int = # the batch size at which resolution gradient steps will be taken
EPOCHS = # number of epochs to train for

MODEL_DIM = # the dimension of the model
MODEL_N_HEADS = # number of attention heads in the model
MODEL_N_LAYERS = # number of transformer layers in the model

OUTPUT_DIR = # the directory to output the training metadata and model weights
```

Refer to the slurm scripts in the top level directory for running training using the slurm workload manager.

Note: in order to run check the slurm scripts, you will need to change the source path to the cloned llama respository. Alternatively you can simply run from the base repository llama directory and run using `torchrun`.

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

Look at [GENERATION.md](GENERATION.md).

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
