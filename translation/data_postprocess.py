import sys
from datasets import DatasetDict, Dataset

# get terminal parameters
args = sys.argv

dataset_dict = {
    'qid': [],
    'question': [],
    'response_j': [],
    'response_k': []
}

for column in dataset_dict.keys():
    # read data from file and save as a list
    with open(f'./translation/data/{args[1]}/{column}.translated', 'r') as f:
        data = f.read().splitlines()

    dataset_dict[column] = data

# create a dataset from the dictionary
dataset = Dataset.from_dict(dataset_dict)

# turn the dataset into a dataset dict
dataset_dict = DatasetDict({'train': dataset})

# push the dataset dict to the hub
dataset_dict.push_to_hub(f'{args[2]}_translated')