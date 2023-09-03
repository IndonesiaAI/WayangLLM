import datasets
import sys

# get terminal parameters
args = sys.argv

if len(args) < 2:
    raise Exception('Please provide dataset name')

# load dataset
dataset = datasets.load_dataset(args[1], split='train')

columns = ['question', 'response_j', 'response_k']

for column in columns:
    # save all text in the column as a file in the data folder
    with open(f'translation/data/{column}', 'w', encoding='utf-8') as f:
        for text in dataset['train'][column]:
            f.write(text + '\n')