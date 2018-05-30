## Semantic Role Labeling with subword composition

The code for ACL18 paper "Character-Level Models versus Morphology in Semantic Role Labeling" by Gözde Gül Şahin and Mark Steedman.

We provide sample training/testing scripts for different subword units under example_scripts folder.

### Scripts Overview 
Train SRL models on CoNLL-09 SRL training sets and test/evaluate trained models on CoNLL-09 evaluation sets for all languages. 

1. **simple_UNIT.sh**: Trains/tests *base SRL models* for the given subword UNIT. Please check train.py for parameter descriptions.

2. **ensemble_UNIT1_UNIT2_UNITn**: Voting ensemble for the provided pretrained base SRL models (UNIT1, UNIT2, ..., UNITn). 

3. **sg_UNIT1_UNIT2_UNITn**: Trains/tests a stack generalizer model from the predictions of pretrained base SRL models (UNIT1, UNIT2, ..., UNITn). 

### Testing Environment

Language: Python 2.7
CUDA: Cuda compilation tools, release 8.0, V8.0.44
Libraries: PyTorch 0.2.0_3, numpy 1.13.0






