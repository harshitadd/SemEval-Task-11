# Propaganda Dection: Submission to SemEval-2020 Shared Task 11
[PsuedoProp at SemEval-2020 Task 11: Propaganda Span Detection Using BERT-CRF and Ensemble Sentence Level Classifier](https://aclanthology.org/2020.semeval-1.233/)

<img width="831" alt="pseudoprop" src="https://user-images.githubusercontent.com/31439716/204010127-03205fa2-6c30-4467-8504-b9230e8ce40c.png">

We propose a sequential BERT-CRF based Span Identification model where the fine-grained detection is carried out only on the articles that are flagged as containing propaganda by an ensemble SLC model. We propose this setup bearing in mind the practicality of this approach in identifying propaganda spans in the exponentially increasing content base where the fine-tuned analysis of the entire data repository may not be the optimal choice due to its massive computational resource requirements. We present our analysis on different voting ensembles for the SLC model. Our system ranks 14th on the test set and 22nd on the development set and with an F1 score of 0.41 and 0.39 respectively.

## Directory Structure
```
├── bert_lstm_ner.py                :BERT-CRF defintion
├── inference.ipynb                 :Permutations for ensembles
├── roberta_slc.ipynb               :Sequence Level Classification of the sentences as being propaganda or not. 
├── span_detection.ipynb            :Fine-Grained span prediction on the propaganda samples. 
└── terminal_predict.py             :e2e pipeline. 
```

Cite: Aniruddha Chauhan and Harshita Diddee. 2020. PsuedoProp at SemEval-2020 Task 11: Propaganda Span Detection Using BERT-CRF and Ensemble Sentence Level Classifier. In Proceedings of the Fourteenth Workshop on Semantic Evaluation, pages 1779–1785, Barcelona (online).International Committee for Computational Linguistics.
