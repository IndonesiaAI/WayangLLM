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
    
    raw_data = [clean_text(text) for text in tqdm(dataset['question'], desc="Processing raw data")]
    raw_data = list(itertools.chain(*raw_data))
    qid_data = [(str(dataset['qid'][i])) * len(str(text).split('\n')) for i, text in enumerate(dataset[column])]

    df = pd.DataFrame({column: raw_data})
    df.to_csv(f'translation/data/{args[1]}/{column}/raw', index=False, header=False, sep='\n')

    df = pd.DataFrame({'qid': qid_data})
    df.to_csv(f'translation/data/{args[1]}/{column}/qid', index=False, header=False, sep='\n')

    # with open(f'translation/data/{args[1]}/{column}/raw', 'w', encoding='utf-8') as f:
    #     # write text in the dataset in batch
    #     for texts in tqdm(raw_data/1000, desc="Writing raw data"):
    #         f.writelines(texts)
    # with open(f'translation/data/{args[1]}/{column}/qid', 'w', encoding='utf-8') as qid:
    #     # write qid in the dataset in batch
    #     for qids in tqdm(qid_data/1000, desc="Writing qid data"):
    #         qid.writelines(qids)

    # with open(f'translation/data/{args[1]}/{column}/raw', 'w', encoding='utf-8') as f, \
    #     open(f'translation/data/{args[1]}/{column}/qid', 'w', encoding='utf-8') as qid:
    #     for i, text in tqdm(enumerate(dataset[column]), total=len(dataset[column]), desc="Processing raw data"):
    #         cleaned_text = clean_text(text)
    #         f.write(cleaned_text + '\n')
    #         qid_data = str(dataset['qid'][i]) * len(str(text).split('\n'))
    #         qid.write(qid_data + '\n')

    # save all text in the column as a file in the data folder
    # with open(f'translation/data/{args[1]}/{column}/raw', 'w', encoding='utf-8') as f, \
    #     open(f'translation/data/{args[1]}/{column}/qid', 'w', encoding='utf-8') as qid:
    #         # write text in the dataset column to the f and 'qid' column to qid,
    #         # but split the text by new line
    #         for i, text in tqdm(enumerate(dataset[column])):
    #             f.write(clean_text(text) + '\n')
    #             qid.write((str(dataset['qid'][i]) + '\n') * len(str(text).split('\n')))

    # create new file with the same name and add .encoded to the end
    with open(f'translation/data/{args[1]}/{column}/encoded', 'w', encoding='utf-8') as f:
        f.write('')

    # create new file with the same name and add .translated to the end
    with open(f'translation/data/{args[1]}/{column}/translated', 'w', encoding='utf-8') as f:
        f.write('')