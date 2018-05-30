#!/bin/sh

export PYTHONIOENCODING=utf-8
source ../../../bin/activate

gpu=1

slangs=("ger" "spa" "fin" "cat" "tur")
llangs=("German" "Spanish" "Finnish" "Catalan" "Turkish")
dirlangs=("german" "spanish" "fin" "catalan" "turkish")
mkdir /afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/ensemble_exp_sgd/

for id in {0..4}
do
   slang="${slangs[id]}"
   llang="${llangs[id]}"
   dirlang="${dirlangs[id]}"
   mkdir /afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/ensemble_exp_sgd/$slang
   echo "Char-Char3-Oracle - Language "$llang
   python ../ensemble.py \
   -test_file "/afs/inf.ed.ac.uk/user/s16/s1691718/data/conll09-$slang/CoNLL2009-ST-evaluation-$llang.txt" \
   -save_dir1 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/char3" \
   -save_dir2 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/oracle" \
   -save_dir3 "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/$dirlang/char" \
   -ens_save_dir "/afs/inf.ed.ac.uk/group/project/s1691718/acl2018_sgd/ensemble_exp_sgd/$slang/char_char3_oracle_avg" \
   -lang $slang \
   -gpuid $gpu
done
echo "Finished testing vote ensembling"
