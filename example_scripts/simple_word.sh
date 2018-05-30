#!/bin/sh
export PYTHONIOENCODING=utf-8
source ../../../bin/activate

echo "Running word on all data"
mkdir /afs/inf.ed.ac.uk/group/project/s1691718/acl2018/

UNIT=word
gpu=0

slangs=("en" "ger" "spa" "fin" "cat" "cze" "tur")
llangs=("English" "German" "Spanish" "Finnish" "Catalan" "Czech" "Turkish")
dirlangs=("english" "german" "spanish" "fin" "catalan" "czech" "turkish")

for id in {0..6}
do
   slang="${slangs[id]}"
   llang="${llangs[id]}"
   dirlang="${dirlangs[id]}"
   mkdir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018/"$dirlang/$UNIT"_all"
   python ../train.py \
    -train_file "/afs/inf.ed.ac.uk/user/s16/s1691718/data/conll09-$slang/CoNLL2009-ST-$llang-train.txt" \
    -val_file "/afs/inf.ed.ac.uk/user/s16/s1691718/data/conll09-$slang/CoNLL2009-ST-$llang-development.txt" \
    -lang $slang \
    -save_dir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018/"$dirlang/$UNIT"_all" \
    -param_init_type "orthogonal" \
    -init_scale 0.01 \
    -optim "sgd" \
    -grad_clip 2 \
    -dropout 0.5 \
    -learning_rate 1 \
    -decay_rate 0.5 \
    -epochs 20 \
    -sub_rnn_size 200 \
    -sub_num_layers 1 \
    -unit "word" \
    -composition "none" \
    -ngram 0 \
    -char_dim 200 \
    -morph_dim 200 \
    -word_dim 200 \
    -layers 1 \
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
    -test_file "/afs/inf.ed.ac.uk/user/s16/s1691718/data/conll09-$slang/CoNLL2009-ST-evaluation-$llang.txt" \
    -save_dir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018/"$dirlang/$UNIT"_all" \
    -lang $slang \
    -gpuid $gpu
done