from sklearn.externals import joblib
from tika import parser
import rouge.rouge_score as r
import nltk

nltk.download('punkt')
import re

loaded_model = joblib.load('fine_tune_BERT_sentence_classification.pkl')
print(loaded_model)

from xml.etree import ElementTree
from nltk import sent_tokenize,word_tokenize
import os
import pandas as pd
import numpy as np
import json

def create_test_set():
    '''
    create the test set
    :return: the DataFrame that contains the test set
    '''
    xml_files = os.listdir('test_papers')
    list_summaries = list()
    dict_of_sentences = dict()
    for f in xml_files:
        file = ElementTree.parse("test_papers/"+f)
        #file = ElementTree.parse("xml/" + f)
        root = file.getroot()
        #print(f)
        sections = dict()

        for div in file.getiterator(tag="{http://www.tei-c.org/ns/1.0}div"):

            sec = ""
            for head in div.iter('{http://www.tei-c.org/ns/1.0}head'):
                #print(head.text)
                if head.text not in sections:
                    sections[head.text] = ""
                    sec = head.text

            text = ""
            for p in div.iter('{http://www.tei-c.org/ns/1.0}p'):
                #print(p.text)
                text += p.text
            sections[sec] = text
        list_summaries.append(sections)
        dict_of_sentences[f] = sections

    dict_of_all_the_sentences = {
        'sentence':[],
        'previous':[],
        'next':[],
        'section':[],
        'paper':[],
        'length':[]
    }



    sentences_list = ["first sentence", "last sentence"]  # list()

    for pdf, section in dict_of_sentences.items():
        print(pdf)
        for head, text in section.items():
            sentences = sent_tokenize(text)

            for index, sentence in enumerate(sentences):
                # if (index + 1) > len(sentences):
                #   dict_of_all_the_sentences['next'].append("last sentence")
                if sentence in dict_of_all_the_sentences['sentence']:
                    continue

                dict_of_all_the_sentences['paper'].append(pdf)
                dict_of_all_the_sentences['length'].append(len(word_tokenize(sentence)))
                sect = re.sub(r'[^A-Za-z0-9]+', " ", head).lstrip().lower()
                dict_of_all_the_sentences['section'].append(sect)
                if sect not in sentences_list:
                    sentences_list.append(sect)
                if index == 0:
                    sen = re.sub(r'[^A-Za-z0-9]+', " ", sentence).lstrip().lower()
                    dict_of_all_the_sentences['sentence'].append(sen)
                    dict_of_all_the_sentences['previous'].append("first sentence")
                    if sen not in sentences_list:
                        sentences_list.append(sen)

                    if (index + 1) == len(sentences):
                        dict_of_all_the_sentences['next'].append("last sentence")
                    else:
                        next_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index + 1)]).lstrip().lower())
                        dict_of_all_the_sentences['next'].append(next_sen)
                        if next_sen not in sentences_list:
                            sentences_list.append(next_sen)
                elif index > 0:
                    sen = re.sub(r'[^A-Za-z0-9]+', " ", sentence).lstrip().lower()
                    dict_of_all_the_sentences['sentence'].append(sen)
                    if sen not in sentences_list:
                        sentences_list.append(sen)

                    pre_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index - 1)]).lstrip().lower())
                    dict_of_all_the_sentences['previous'].append(pre_sen)
                    if pre_sen not in sentences_list:
                        sentences_list.append(pre_sen)
                    if (index + 1) == len(sentences):
                        dict_of_all_the_sentences['next'].append("last sentence")
                    else:
                        next_sen = (re.sub(r'[^A-Za-z0-9]+', " ", sentences[(index + 1)]).lstrip().lower())
                        dict_of_all_the_sentences['next'].append(next_sen)
                        if next_sen not in sentences_list:
                            sentences_list.append(next_sen)


    test_set = pd.DataFrame(dict_of_all_the_sentences['sentence'])
    test_set.rename(index=str, columns={0: "sentence"},inplace=True)
    test_set['previous'] = dict_of_all_the_sentences['previous']
    test_set['next'] = dict_of_all_the_sentences['next']
    test_set['section'] = dict_of_all_the_sentences['section']
    test_set['length'] = dict_of_all_the_sentences['length']
    test_set['paper'] = dict_of_all_the_sentences['paper']

    #test_set['sentence'] =pd.DataFrame(dict_of_all_the_sentences['sentence'])
    print(test_set)

    return test_set


def extract_embeddings():
    '''
    extract the BERT embedding of the sentences and rhe sections of the test set
    :return:
    '''

    test_set = create_test_set()
    test_sentences_list = ['first sentence','last sentence']
    for row in test_set.iterrows():
        sentence = str(row[1][0])
        section = str(row[1][3])
        if sentence not in test_sentences_list:
            test_sentences_list.append(sentence)
        if section not in test_sentences_list:
            test_sentences_list.append(section)


    with open('test_sentences_list.txt','w') as f:
        for line in test_sentences_list:
            f.write("%s\n"%(line))

    os.system('python extract_features.py --input_file=C:/Users/user/PycharmProjects/bert/test_sentences_list.txt --output_file=test_output_layer_-3.json --vocab_file=D:/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=D:/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=C:/Users/user/PycharmProjects/bert/my_dataset_output/model.ckpt-3000  --layers=-3  --max_seq_length=128 --batch_size=8')


def get_embeddings():

    sentences = dict()
    with open('test_sentences_list.txt','r') as f:
        for index,line in enumerate(f):
            sentences[index] = line.strip()


    embeddings = dict()
    with open('C:/Users/user/PycharmProjects/bert/test_output_layer_-3.json', 'r',encoding='utf-8') as f:
        for line in f:
            embeddings[json.loads(line)['linex_index']] = np.asarray(json.loads(line)['features'])

    sentence_emb = dict()

    for key,value in sentences.items():
        sentence_emb[value] = embeddings[key]


    return sentence_emb

