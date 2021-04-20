# encoding=utf-8

"""
基于命令行的在线预测方法
@Author: Macan (ma_cancan@163.com) 
"""

import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
from zipfile import ZipFile
args = get_args_parser()


# /content/BERT-BiLSTM-CRF-NER/output/checkpoint
# model_dir = r'../../output'

model_dir = '/content/BERT-BiLSTM-CRF-NER/output/'
bert_dir = '/content/BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12'
# dev_folder='/content/dev-articles.zip'
is_training=False
use_one_hot_embeddings=False
batch_size=1

## PREPARING TEST DATA 
# with ZipFile(dev_folder, 'r') as zipObj:
#    zipObj.extractall()

# dev_folder='./dev-articles/'

sentences_test=[]
index_test=[]
article_id_list_test = []

def createDataFrame_test(data):
    test_files=os.listdir(data)
    for file in test_files:
        with open(os.path.join(data,file),encoding="utf8") as f:
            article_id=file.split('.')[0].split('e')[1]
            text=f.read()
            tokenizer=PunktSentenceTokenizer()
            sentence_main=tokenizer.span_tokenize(text)
            sentences=tokenizer.tokenize(text)
            sentence_main=[span[0] for span in sentence_main]
            
            sentences_test.append(sentences)
            index_test.append(sentence_main)
            article_id_list_test.append(article_id)

    sentences_test_main = []
    index_test_main = []
    article_id_list_test_final = []

    for i,b in zip(sentences_test,article_id_list_test):
        for line in i:
            sentences_test_main.append(line)
            article_id_list_test_final.append(b)

    for j in index_test:
        for line in j:
            index_test_main.append(line)


    return article_id_list_test_final,sentences_test_main,index_test_main 



##
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_online():
    final_pred =[]
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    df_dev_all_prop = pd.read_csv('/content/BERT-BiLSTM-CRF-NER/dev_prop_pred_startc.csv')
    # ids, sentences, indices = createDataFrame_test(dev_folder)
    ids, sentences, indices = df_dev_all_prop['article_id'], df_dev_all_prop['sentences'], df_dev_all_prop['startC']

    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids
    
    global graph
    with graph.as_default():
        print(id2label)
        ctr=0
        for i,sentence,index in zip(ids,sentences,indices):
            ctr+=1
            temp_sent = sentence
            sentence_token = sentence.split() 
            # if(len(sentence_token)>128):
                  # p1 = ' '.join(sentence_token[0:128])
                  # print(p1)
                  # sentence = tokenizer.tokenize(p1)
                  # input_ids, input_mask, segment_ids, label_ids = convert(sentence)
                  # feed_dict = {input_ids_p: input_ids,input_mask_p: input_mask}
                  # pred_ids_result = sess.run([pred_ids], feed_dict)
                  # pred_label_result = convert_id_to_label(pred_ids_result, id2label)
                  # l=[]
                  # l.append(i)
                  # l.append(p1)
                  # l.append(pred_label_result)
                  # final_pred.append(l)

                  # p2 = ' '.join(sentence_token[128:])
                  # print(p2)
                  # sentence = tokenizer.tokenize(p2)
                  # input_ids, input_mask, segment_ids, label_ids = convert(sentence)
                  # feed_dict = {input_ids_p: input_ids,input_mask_p: input_mask}
                  # pred_ids_result = sess.run([pred_ids], feed_dict)
                  # pred_label_result = convert_id_to_label(pred_ids_result, id2label)
                  # l=[]
                  # l.append(i)
                  # l.append(p2)
                  # l.append(pred_label_result)
                  # final_pred.append(l)
            # else: 
            sentence = tokenizer.tokenize(sentence)
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)
            feed_dict = {input_ids_p: input_ids,
                          input_mask_p: input_mask}
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            l=[]
            l.append(i)
            l.append(temp_sent)
            l.append(sentence)
            l.append(pred_label_result)
            l.append(index)
            final_pred.append(l)
            print(ctr," ",len(sentence)," ",len(pred_label_result[0]))

    with open('/content/pred_maker_without_slicing.bin','wb') as file:
      pickle.dump(final_pred,file)        
            
  

# 
            # print(pred_label_result)
            # print("Length if pred_label_result: ", len(pred_label_result[0]))
            # print("Orignal sentence: ",orig_sent.split())
            # print("Length of input tokens: ", len(orig_sent.split()))
            # print("Tokenized sentence: ", sentence)
            # print("Length of tokenized sentence: ", len(sentence))
            # #todo: 组合策略
            # result = strage_combined_link_org_loc(sentence, pred_label_result[0])
            # print('time used: {} sec'.format((datetime.now() - start).total_seconds()))
def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result



def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        print(', '.join(line))

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    print_output(loc, 'LOC')
    print_output(person, 'PER')
    print_output(org, 'ORG')


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    predict_online()

