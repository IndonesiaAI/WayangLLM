import datasets
import sys
import os
import re

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

def clean_text(text) -> list:
    text = text.replace('\n', '[ENT]')
    sentences = nltk.sent_tokenize(text)
    # split sentences with : ! ? ,
    sentences = [
        re.sub(r'([,:!?])', r'\\1[CUT]', sentence).split('[CUT]')
        for sentence in sentences]
    return sentences

for column in columns:
    os.mkdir(f'translation/data/{args[1]}/{column}')

    raw_data, qid_data = [], []
    for text_, qid_ in tqdm(zip(dataset[column], dataset['qid']), desc="Processing raw data"):
        cleaned = clean_text(text_)
        raw_data += cleaned
        qid_data += [qid_] * len(cleaned)

    df = pd.DataFrame({column: raw_data})
    df.to_csv(f'translation/data/{args[1]}/{column}/raw', index=False, header=False, sep='\n')

    df = pd.DataFrame({'qid': qid_data})
    df.to_csv(f'translation/data/{args[1]}/{column}/qid', index=False, header=False, sep='\n')

    # create new file with the same name and add .encoded to the end
    with open(f'translation/data/{args[1]}/{column}/encoded', 'w', encoding='utf-8') as f:
        f.write('')

    # create new file with the same name and add .translated to the end
    with open(f'translation/data/{args[1]}/{column}/translated', 'w', encoding='utf-8') as f:
        f.write('')


with open(f'translation/data/{args[1]}/qid', 'w', encoding='utf-8') as f:
    for qid in dataset['qid']:
        f.write(str(qid) + '\n')