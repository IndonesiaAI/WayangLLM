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
    with open(f'./translation/data/{args[1]}/{column}/translated_decoded', 'r') as f:
        data = f.read().splitlines()

        with open(f'./translation/data/{args[1]}/{column}/qid', 'r') as qid:
            qid = qid.read().splitlines()

        df = pd.DataFrame({'qid': qid, column: data})
        # filter data with qid not in dataset_dict['qid']
        df = df[df['qid'].isin(dataset_dict['qid'])]
        df = df.groupby('qid').groups
        df = [' '.join(df[qid_]) for qid_ in tqdm(df.keys())]
        dataset_dict[column] = df

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