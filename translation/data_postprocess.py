import sys
from datasets import DatasetDict, Dataset
from tqdm import tqdm
import pandas as pd

# get terminal parameters
args = sys.argv

dataset_dict = {
    'qid': [],
    'question': [],
    'response_j': [],
    'response_k': []
}

with open(f'./translation/data/{args[1]}/qid', 'r') as f:
    data = f.read().splitlines()

dataset_dict['qid'] = data

for column in list(dataset_dict.keys())[1:]:
    # read data from file and save as a list
    with open(f'./translation/data/{args[1]}/{column}/translated.decoded', 'r') as f:
        data = f.read().splitlines()

    with open(f'./translation/data/{args[1]}/{column}/index', 'r') as f:
        index = f.read().splitlines()

    for i, j in tqdm(zip(index[:-1], index[1:])):
        dataset_dict[column].append(' '.join(data[int(i):int(j)]))

    # # iterate through each dataset_dict['qid'] and
    # # find all the data in zip(data, qid) that has the same qid
    # # then combine the data into a string and save it to dataset_dict[column]
    # dataset_dict[column] = [''.join([d[0] for d in zip(data, qid) if d[1] == qid_]) for qid_ in tqdm(dataset_dict['qid'])]

# create a dataset from the dictionary
dataset = Dataset.from_dict(dataset_dict)

# turn the dataset into a dataset dict
dataset_dict = DatasetDict({'train': dataset})

# push the dataset dict to the hub
dataset_dict.push_to_hub(f'IndonesiaAI/{args[2]}_translated')