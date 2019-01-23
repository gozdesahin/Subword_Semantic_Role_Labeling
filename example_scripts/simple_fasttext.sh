#!/bin/sh
export PYTHONIOENCODING=utf-8
source activate srlexp

echo "Running fasttext on all data"
mkdir /home/sahin/Workspace/results/srlexp/fasttext

UNIT=word
gpu=0

slangs=("ger" "spa" "fin" "tur")
llangs=("German" "Spanish" "Finnish" "Turkish")
dirlangs=("german" "spanish" "fin" "turkish")
emblangs=("de" "es" "fi" "tr")

for id in {0..4}
do
   slang="${slangs[id]}"
   llang="${llangs[id]}"
   dirlang="${dirlangs[id]}"
   emblang="${emblangs[id]}"
   mkdir "/home/sahin/Workspace/results/srlexp/fasttext/"$dirlang
   python ../train.py \
    -train_file "/home/sahin/Workspace/dataset/conll09-$slang/CoNLL2009-ST-$llang-train.txt" \
    -val_file "/home/sahin/Workspace/dataset/conll09-$slang/CoNLL2009-ST-$llang-development.txt" \
    -lang $slang \
    -save_dir "/home/sahin/Workspace/results/srlexp/fasttext/"$dirlang \
    -param_init_type "orthogonal" \
    -init_scale 0.01 \
    -optim "sgd" \
    -grad_clip 2 \
    -dropout 0.5 \
    -learning_rate 1 \
    -decay_rate 0.5 \
    -epochs 20 \
    -unit "word" \
    -pre_word_vecs "/home/sahin/Workspace/embeddings/wiki."$emblang/"wiki."$emblang \
    -fixed_embed \
    -w2vtype "fasttext" \
    -composition "none" \
    -ngram 0 \
    -word_dim 300 \
    -layers 2 \
    -numdir 2 \
    -hidden_size 200 \
    -wp 1 \
    -use_region_mark False \
    -use_binary_mask True \
    -batch_size 32 \
    -max_seq_length 200 \
    -cont "false" \
    -gpuid $gpu
    echo "test word"
    python ../test.py \
    -test_file "/home/sahin/Workspace/dataset/conll09-$slang/CoNLL2009-ST-evaluation-$llang.txt" \
    -save_dir "/home/sahin/Workspace/results/srlexp/fasttext/"$dirlang \
    -lang $slang \
    -gpuid $gpu
done