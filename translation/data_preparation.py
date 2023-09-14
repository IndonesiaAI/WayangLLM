import datasets
import sys
import os

# get terminal parameters
args = sys.argv

if len(args) < 2:
    raise Exception('Please provide dataset name')

if len(args) > 2:
    raise Exception('Too many arguments')

# load dataset
dataset = datasets.load_dataset(args[1], split='train')

columns = ['qid', 'question', 'response_j', 'response_k']

# check if translation/data folder exists
if not os.path.exists('translation/data'):
    # make a new folder for the dataset
    os.mkdir('translation/data')

# make a new folder for the dataset
os.mkdir(f'translation/data/{args[1]}')

for column in columns:
    # save all text in the column as a file in the data folder
    os.mkdir(f'translation/data/{args[1]}/{column}')

    with open(f'translation/data/{args[1]}/{column}', 'w', encoding='utf-8') as f:
        for text in dataset['train'][column]:
            f.write(text + '\n')