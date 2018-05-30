#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
source ../../../bin/activate

gpu=0

slangs=("ger" "spa" "fin" "cat" "tur")
llangs=("German" "Spanish" "Finnish" "Catalan" "Turkish")
dirlangs=("german" "spanish" "fin" "catalan" "turkish")

for id in {0..4}
do
   slang="${slangs[id]}"
   llang="${llangs[id]}"
   dirlang="${dirlangs[id]}"
   echo "Train - SG - Char-Char3 - Language "$llang
   python ../train_stack_gen.py \
    -val_file "/afs/inf.ed.ac.uk/user/s16/s1691718/data/conll09-$slang/CoNLL2009-ST-$llang-development.txt" \
    -save_dir1 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/char" \
    -save_dir2 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/char3" \
    -save_dir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/ensemble_exp_sgd/$slang/char_char3_sg" \
    -param_init_type "orthogonal" \
    -optim "adam" \
    -learning_rate 0.02 \
    -epochs 25 \
    -indim 2 \
    -hiddim 64 \
    -lang $slang \
    -gpuid $gpu
   echo "Test - SG - Char-Char3 - Language "$llang
   python ../test_stack_gen.py \
    -test_file "/afs/inf.ed.ac.uk/user/s16/s1691718/data/conll09-$slang/CoNLL2009-ST-evaluation-$llang.txt" \
    -save_dir1 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/char" \
    -save_dir2 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/char3" \
    -ens_model_dir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/ensemble_exp_sgd/$slang/char_char3_sg" \
    -ens_save_dir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/ensemble_exp_sgd/$slang/char_char3_sg" \
    -lang $slang \
    -gpuid $gpu
done




