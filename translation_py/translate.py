from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset

import sys

# get terminal parameters
args = sys.argv

# Load model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-id'

print('Loading model and tokenizer...')
print('Model name:', model_name)
print('Dataset name:', args[1])
print('Batch size:', args[2])
print('Repo target:', args[3])
print('---------------------------------')
print('')

print('Loading model and tokenizer...')
print('')

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# load model to gpu
model.to('cuda')

print('loading dataset...')
print('')

# Load dataset
dataset = load_dataset(args[1], split='train')

def translate_batch(batch, column_name: str):
    # Tokenize input
    src_text = batch[column_name]
    # split the text inside the list by each sentence
    src_text = [text.split('.') for text in src_text]

    # tokenize the text
    batch = tokenizer(src_text, return_tensors="pt")

    # Translate
    gen = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(gen, skip_special_tokens=True)

    batch[column_name] = tgt_text
    return batch

print('Translating...')
print('---------------------------------')

print('Translating question...')
print('')

# Translate
dataset['train']['question'] = dataset['train']['question'].apply(lambda batch: translate_batch(batch, 'question'), batched=True, batch_size=args[2])

print('Translating response_j...')
print('')

dataset['train']['response_j'] = dataset['train']['response_j'].apply(lambda batch: translate_batch(batch, 'response_j'), batched=True, batch_size=args[2])

print('Translating response_k...')
print('')

dataset['train']['response_k'] = dataset['train']['response_k'].apply(lambda batch: translate_batch(batch, 'response_k'), batched=True, batch_size=args[2])

print('Pushing dataset...')
print('')

# push dataset
dataset.push_to_hub(args[3])