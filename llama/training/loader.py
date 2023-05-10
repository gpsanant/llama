import json
# our training data looks like this:
# {"text": "This is a sentence."}
# {"text": "This is another sentence."}
# one json object per line
# We should convert this to tokens using tokenzer.encode() with bos=True and eos=True
# and keep data as each list of tokens from the beginning of the sentence to every token in the sentence
# and the labels as a 1 hot encoded vector of the next token in the sentence
# We should load the entire file adding each peice of data to the dataset using the above method
# We should then create a dataloader using the dataset
def load_data(file_path, tokenizer):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            tokens = tokenizer.encode(text, bos=True, eos=True)
            one_hots = torch.nn.functional.one_hot(tokens, num_classes=tokenizer.n_words)
            for i in range(1, len(tokens)):
                # data is all tokens up to the current token
                # label is a 1 hot encoding of the current token
                dataset.append((tokens[:i], one_hots[i]))
    return dataset