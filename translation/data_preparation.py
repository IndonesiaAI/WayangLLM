import datasets
import sys
import os
import re

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
    # Insert newlines after sentence enders
    text = re.sub('([.?!;]) ', r'\1\n', text)
    return text.replace('\n\n', '\n')

for column in columns:
    os.mkdir(f'translation/data/{args[1]}/{column}')
    
    raw_data = [clean_text(text) for text in tqdm(dataset[column], desc="Processing raw data")]
    qid_data = [(str(dataset['qid'][i])) * len(str(text).split('\n')) for i, text in enumerate(dataset[column])]

    with open(f'translation/data/{args[1]}/{column}/raw', 'w', encoding='utf-8') as f:
        f.writelines(raw_data)
    with open(f'translation/data/{args[1]}/{column}/qid', 'w', encoding='utf-8') as qid:
        qid.writelines(qid_data)

    # save all text in the column as a file in the data folder
    # with open(f'translation/data/{args[1]}/{column}/raw', 'w', encoding='utf-8') as f:
    #     with open(f'translation/data/{args[1]}/{column}/qid', 'w', encoding='utf-8') as qid:
    #         # write text in the dataset column to the f and 'qid' column to qid,
    #         # but split the text by new line
    #         for i, text in tqdm(enumerate(dataset[column])):
    #             f.write(str(text) + '\n')
    #             qid.write((str(dataset['qid'][i]) + '\n') * len(str(text).split('\n')))

    # create new file with the same name and add .encoded to the end
    with open(f'translation/data/{args[1]}/{column}/encoded', 'w', encoding='utf-8') as f:
        f.write('')

    # create new file with the same name and add .translated to the end
    with open(f'translation/data/{args[1]}/{column}/translated', 'w', encoding='utf-8') as f:
        f.write('')