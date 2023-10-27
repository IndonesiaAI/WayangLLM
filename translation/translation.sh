#!/bin/bash

# Prepare data
python ./data_preparation.py $1 :

for column in question response_j response_k;
do
    # Encode data
    ./model/preprocess.sh id ./model/source.spm < \
        ./translation/data/$1/$column/raw > \
        ./translation/data/$1/$column/encoded

    # Translate
    cat ./translation/data/$1/$column/encoded | \
        ./marian/build/marian-decoder \
        -c ./model/decoder.yml \
        -d 0 \
        --mini-batch 64 \
        --maxi-batch-sort src \
        --maxi-batch 100 \
        --optimize \
        -w 2500 > \
        ./translation/data/$1/$column/translated
done

# postprocess data
python ./data_postprocess.py $1 $2