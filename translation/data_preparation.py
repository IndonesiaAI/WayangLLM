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

for column in columns:
    os.mkdir(f'translation/data/{args[1]}/{column}')
    
    # save all text in the column as a file in the data folder
    with open(f'translation/data/{args[1]}/{column}/raw', 'w', encoding='utf-8') as f:
        with open(f'translation/data/{args[1]}/{column}/qid', 'w', encoding='utf-8') as qid:
            # write text in the dataset column to the f and 'qid' column to qid,
            # but split the text by new line
            for i, text in enumerate(dataset[column]):
                for line in str(text).split('\n'):
                    f.write(line + '\n')
                    qid.write(str(dataset['qid'][i]) + '\n')

    # create new file with the same name and add .encoded to the end
    with open(f'translation/data/{args[1]}/{column}/encoded', 'w', encoding='utf-8') as f:
        f.write('')

    # create new file with the same name and add .translated to the end
    with open(f'translation/data/{args[1]}/{column}/translated', 'w', encoding='utf-8') as f:
        f.write('')