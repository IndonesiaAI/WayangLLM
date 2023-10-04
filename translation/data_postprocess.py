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

with open(f'./translation/data/{args[1]}/qid', 'r') as f:
    data = f.read().splitlines()

dataset_dict['qid'] = data

for column in dataset_dict.keys()[1:]:
    # read data from file and save as a list
    with open(f'./translation/data/{args[1]}/{column}/translated', 'r') as f:
        data = f.read().splitlines()

        with open(f'./translation/data/{args[1]}/{column}/qid', 'r') as qid:
            qid = qid.read().splitlines()
            
        # iterate through each dataset_dict['qid'] and
        # find all the data in zip(data, qid) that has the same qid
        # then combine the data into a string and save it to dataset_dict[column]
        index = 0
        for qid_ in dataset_dict['qid']:
            temp = ""
            for i in range(index, len(qid)):
                if qid[i] == qid_:
                    temp += data[i] + ' '
                else:
                    index = i
                    break
            dataset_dict[column].append(temp)

# create a dataset from the dictionary
dataset = Dataset.from_dict(dataset_dict)

# turn the dataset into a dataset dict
dataset_dict = DatasetDict({'train': dataset})

# push the dataset dict to the hub
dataset_dict.push_to_hub(f'IndonesiaAI/{args[2]}_translated')