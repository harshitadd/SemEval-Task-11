# Propaganda Dection: Submission to SemEval-2020 Shared Task 11
[PsuedoProp at SemEval-2020 Task 11: Propaganda Span Detection Using BERT-CRF and Ensemble Sentence Level Classifier](https://aclanthology.org/2020.semeval-1.233/)

## Directory Structure
```
├── bert_lstm_ner.py                :BERT-CRF defintion
├── inference.ipynb                 :Permutations for ensembles
├── roberta_slc.ipynb               :Sequence Level Classification of the sentences as being propaganda or not. 
├── span_detection.ipynb            :Fine-Grained span prediction on the propaganda samples. 
└── terminal_predict.py             :e2e pipeline. 
```

Cite: Aniruddha Chauhan and Harshita Diddee. 2020. PsuedoProp at SemEval-2020 Task 11: Propaganda Span Detection Using BERT-CRF and Ensemble Sentence Level Classifier. In Proceedings of the Fourteenth Workshop on Semantic Evaluation, pages 1779–1785, Barcelona (online).International Committee for Computational Linguistics.