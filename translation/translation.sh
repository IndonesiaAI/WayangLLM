#!/bin/bash

# Prepare data
python ./data_preparation.py $1

for column in question response_j response_k;
do
    # Encode data
    cat ./data/$1/$column/raw | ./marian/build/spm_encode \
        --model=./model/target.spm \
        > ./translation/data/$1/$column/encoded

    # Translate
    cat ./data/$1/$column/encoded | ./marian/build/marian-decoder \
        -c ./model/decoder.yml \
        -d 0 \
        -b 6 \
        --normalize 0.6 \
        --mini-batch 64 \
        --maxi-batch-sort src \
        --maxi-batch 100 \
        -w 2500 \
        > ./data/$1/$column/translated
done

# postprocess data
python ./data_postprocess.py $1 $2