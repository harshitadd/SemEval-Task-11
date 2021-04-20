# SemEval
File for Span Identification:

Final_BERT_NER_LSTM (4).ipynb – This file is used for Span Identification. Running it without GPU will take a lot of time. The code has been written to run on Google Colab, GitHub repositories have to be cloned, their code is present in the notebook itself.

Once the repo is cloned, in the BERT-BiLSTM-CRF-NER folder

Replace bert_lstm_ner.py and train_helper.py in BERT-BiLSTM-CRF-NER/bert_base/train/ folder with bert_lstm_ner.py and train_helper.py file in the OneDrive folder

Replace terminal_predict.py in BERT-BiLSTM-CRF-NER/ folder with terminal_predict.py in OneDrive folder

After unzipping the uncased_L-12_H-768_A-12.zip file create NERdata folder in BERT-BiLSTM-CRF-NER folder and add train.txt, dev.txt and test.txt files to it.

Replace bert_lstm_ner.py and train_helper.py in /usr/local/lib/python3.6/dist-packages/bert_base-0.0.9-py3.6.egg/bert_base/train/ folder with bert_lstm_ner.py and train_helper.py file in the OneDrive folder

Add dev_prop_pred_startc.csv to BERT-BiLSTM-CRF-NER folder

Run

!bert-base-ner-train \

-data_dir /content/BERT-BiLSTM-CRF-NER/NERdata \

-output_dir /content/BERT-BiLSTM-CRF-NER/output\

-init_checkpoint /content/BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12/bert_model.ckpt \

-bert_config_file /content/BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12/bert_config.json \

-vocab_file /content/BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12/vocab.txt

The last command to run is !python terminal_predict.py

After running terminal_predict.py a file named pred_maker_without_slicing.bin in the /content/ folder in Colab will be created

Creating the prediction file:

Run Making_pred_(2).ipynb file which uses the pred_maker_without_slicing.bin file to create the prediction file in the format that is requested by the task organizers to evaluat
