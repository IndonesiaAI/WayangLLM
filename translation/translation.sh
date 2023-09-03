!#/bin/bash

# Encode source file
cat $1 | ./marian/build/spm_encode --model=./model/source.spm > $1.encoded

# Translate
cat $1.encoded | ./marian/build/marian-decoder -c ./model/decoder.yml -d 0 -b 6 --normalize 0.6 --mini-batch 64 --maxi-batch-sort --maxi-batch 100 -w 2500 > $1.translated