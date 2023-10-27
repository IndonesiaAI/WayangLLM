import datasets
import sys
import os
import re
from itertools import chain
import json
import nltk
import pandas as pd

from tqdm import tqdm

# get terminal parameters
args = sys.argv

nltk.download('punkt')

if len(args) < 2:
    raise Exception('Please provide dataset name')

# load dataset
dataset = datasets.load_dataset(f"IndonesiaAI/{args[1]}", split=f'train[{args[2]}]')

columns = ['question', 'response_j', 'response_k']

# check if translation/data folder exists
if not os.path.exists('translation/data'):
    # make a new folder for the dataset
    os.mkdir('translation/data')

# make a new folder for the dataset
os.mkdir(f'translation/data/{args[1]}')

split_punctuation = re.compile(r'(?<=[\.\,\!\?\:\_\=\;])(?=\s)')
split_quote = re.compile(r'([^\w]\".*?\"[^\w]|[^\w]\'.*?\'[^\w]|[^\w]\(.*?\)[^\w]|[^\w]\[.*?\][^\w])')
sub_repetition = re.compile(r'([^\w\s])\1{3,}')
split_ent = re.compile(r'(?<=\[ENT\])\s*')

def clean_text(text) -> list:
    text = text.replace('\n', '[ENT]')
    sentences = split_quote.split(text)

    processed_sentences = (
        split_ent.split(
            sub_repetition.sub(' ', sentence)
        ) for sentence in chain.from_iterable(split_punctuation.split(sentence) for sentence in sentences)
    )

    processed_sentences = list(chain.from_iterable(processed_sentences))

    if any(len(sentence) > 768 for sentence in processed_sentences):
        return ['[EMP]']

    #processed_sentences = [repr(sentence) for sentence in processed_sentences]

    return processed_sentences

def clean_function(examples):
    """modify the function to work with dataset.map"""
    for column in columns:
        cleaned_data = []
        for text in examples[column]:
            cleaned_data.append(clean_text(text))
        examples[column] = cleaned_data
    return examples

# using dataset.map with batched=True to clean the dataset
dataset = dataset.map(clean_function, batched=True, batch_size=1000)

# Now, save the cleaned columns
for column in columns:
    os.mkdir(f'translation/data/{args[1]}/{column}')

    raw_data = [item for sublist in dataset[column] for item in sublist]
    df = pd.DataFrame({column: raw_data})
    df.to_csv(f'translation/data/{args[1]}/{column}/raw', index=False, header=False, sep='\n')

    indexes = [0]
    for sublist in tqdm(dataset[column], desc="Processing raw data"):
        indexes += [indexes[-1] + len(sublist)]

    df = pd.DataFrame({'index': indexes})
    df.to_csv(f'translation/data/{args[1]}/{column}/index', index=False, header=False, sep='\n')

    # create new file with the same name and add .encoded to the end
    with open(f'translation/data/{args[1]}/{column}/encoded', 'w', encoding='utf-8') as f:
        f.write('')

    # create new file with the same name and add .translated to the end
    with open(f'translation/data/{args[1]}/{column}/translated', 'w', encoding='utf-8') as f:
        f.write('')

with open(f'translation/data/{args[1]}/qid', 'w', encoding='utf-8') as f:
    for qid in dataset['qid']:
        f.write(str(qid) + '\n')