import datasets
import sys
import os
import re
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm

# get terminal parameters
args = sys.argv

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

with open(f'translation/data/{args[1]}/qid', 'w', encoding='utf-8') as f:
    for qid in dataset['qid']:
        f.write(str(qid) + '\n')

def clean_text(text):
    text = str(text)
    # Remove unwanted characters
    pattern = re.compile(r'[^\w\s.,:;?!-]')
    text = re.sub(pattern, '', text)
    # split after sentence enders
    pattern = re.compile(r'[.,:;?!\n]+')
    text = re.split(pattern, text)
    return text

for column in columns:
    os.mkdir(f'translation/data/{args[1]}/{column}')
    
    raw_data, qid_data = [], []
    for question_, qid_ in tqdm(zip(dataset['question'], dataset['qid']), desc="Processing raw data"):
        raw_data += clean_text(question_)
        qid_data += [qid_] * len(raw_data[-1])

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