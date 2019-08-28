import argparse

from sklearn.externals import joblib
import rouge.rouge_score as r
import nltk
nltk.download('punkt')
import re
from xml.etree import ElementTree
from nltk import sent_tokenize,word_tokenize
import os
import pandas as pd
import numpy as np
import json
from numpy import dot
from numpy.linalg import norm
from tika import parser

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

    print(test_set)

    return test_set


def extract_embeddings(input,output,vocab,config,check,):
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

    #os.system('python extract_features.py --input_file=C:/Users/user/PycharmProjects/bert/test_sentences_list.txt --output_file=test_output_layer_-3.json --vocab_file=D:/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=D:/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=C:/Users/user/PycharmProjects/bert/my_dataset_output/model.ckpt-3000  --layers=-3  --max_seq_length=128 --batch_size=8')

    os.system('python extract_features.py '+
              '--input_file='+input+
              ' --output_file='+output+
              ' --vocab_file='+vocab+
              ' --bert_config_file='+config+
              ' --init_checkpoint='+check+
              ' --layers=-3  --max_seq_length=128 --batch_size=8')


def get_embeddings(embeddings_file):
    '''
    Stores the embeddings of each sentence and section of the test set in a dictionary
    :return: embeddings of each sentence
    '''
    sentences = dict()
    with open('test_sentences_list.txt','r') as f:
        for index,line in enumerate(f):
            sentences[index] = line.strip()


    embeddings = dict()
    with open(embeddings_file, 'r',encoding='utf-8') as f:
        for line in f:
            embeddings[json.loads(line)['linex_index']] = np.asarray(json.loads(line)['features'])

    sentence_emb = dict()

    for key,value in sentences.items():
        sentence_emb[value] = embeddings[key]


    return sentence_emb

def first_approach(embeddings_file):

    bert_emb = get_embeddings(embeddings_file)
    #loaded_model = joblib.load('fine_tune_BERT_sentence_classification.pkl')
    loaded_model = joblib.load('summarizer.pkl')

    test_set = create_test_set()
    papers = os.listdir("test_papers")
    summaries = dict()
    words_of_summaries = list()
    sentences_of_summaries = list()

    for paper in papers:
        print(paper)
        if paper.__eq__("250.tei.xml"):
            continue
        if paper.__eq__("289.tei.xml"):
            continue

        check_me = test_set[test_set['paper'].__eq__(paper)]
        length = list()
        sentence_emb = list()
        previous_emb = list()
        section_emb = list()
        next_list = list()
        for row in check_me.iterrows():
            sentence = row[1][0].strip()
            previous = row[1][1].strip()
            nexts = row[1][2].strip()
            section = row[1][3].strip()

            if sentence in bert_emb:
                sentence_emb.append(bert_emb[sentence])
            else:
                sentence_emb.append(np.zeros(768))

            if previous in bert_emb:
                previous_emb.append(bert_emb[previous])
            else:
                previous_emb.append(np.zeros(768))

            if nexts in bert_emb:
                next_list.append(bert_emb[nexts])
            else:
                next_list.append(np.zeros(768))

            if section in bert_emb:
                section_emb.append(bert_emb[section])
            else:
                section_emb.append(np.zeros(768))

            length.append(row[1][4])

        summary_text = ""
        sentence_emb1 = np.asarray(sentence_emb)
        next_emb = np.asarray(next_list)
        previous_emb = np.asarray(previous_emb)
        section_emb = np.asarray(section_emb)
        length = np.asarray(length)
        features = np.concatenate([sentence_emb, previous_emb, next_emb, section_emb], axis=1)
        features = np.column_stack([features, length])
        predictions = loaded_model.predict(features)
        no_sen = 0
        words = 0
        for index, pred in enumerate(predictions):
            if pred == 1:
                summary_text += check_me.iloc[index, 0] + " "

                no_sen += 1
        words += len(word_tokenize(summary_text))
        words_of_summaries.append(words)
        sentences_of_summaries.append(no_sen)
        summaries[paper] = summary_text

    print("Number of words in summary using Logistic Regression as classifier:", np.mean(words_of_summaries))
    print("Number of sentences in summary using Logistic Regression as classifier:",
              np.mean(sentences_of_summaries))

    return summaries


def second_approach(embeddings_file):

    bert_emb = get_embeddings(embeddings_file)
    #loaded_model = joblib.load('fine_tune_BERT_sentence_classification.pkl')
    loaded_model = joblib.load('summarizer.pkl')

    test_set = create_test_set()
    papers = os.listdir("test_papers")
    summaries_60 = dict()
    words_of_summaries_60 = list()
    sentences_of_summaries_60 = list()


    summaries_70 = dict()
    words_of_summaries_70 = list()
    sentences_of_summaries_70 = list()


    summaries_80 = dict()
    words_of_summaries_80 = list()
    sentences_of_summaries_80 = list()


    summaries_90 = dict()
    words_of_summaries_90 = list()
    sentences_of_summaries_90 = list()


    for paper in papers:
        print(paper)
        if paper.__eq__("250.tei.xml"):
            continue
        if paper.__eq__("289.tei.xml"):
            continue


        check_me = test_set[test_set['paper'].__eq__(paper)]
        length = list()
        sentence_emb = list()
        previous_emb = list()
        section_emb = list()
        next_list = list()
        jj_redick = 0
        for row in check_me.iterrows():
            sentence = row[1][0].strip()
            previous = row[1][1].strip()
            nexts = row[1][2].strip()
            section = row[1][3].strip()

            if sentence in bert_emb:
                sentence_emb.append(bert_emb[sentence])
            else:
                sentence_emb.append(np.zeros(768))
                jj_redick += 1

            if previous in bert_emb:
                previous_emb.append(bert_emb[previous])
            else:
                previous_emb.append(np.zeros(768))

            if nexts in bert_emb:
                next_list.append(bert_emb[nexts])
            else:
                next_list.append(np.zeros(768))

            if section in bert_emb:
                section_emb.append(bert_emb[section])
            else:
                section_emb.append(np.zeros(768))

            length.append(row[1][4])

        summary_text_60 = ""
        summary_text_70 = ""
        summary_text_80 = ""
        summary_text_90 = ""

        #sentence_emb1 = np.asarray(sentence_emb)
        next_emb = np.asarray(next_list)
        previous_emb = np.asarray(previous_emb)
        section_emb = np.asarray(section_emb)
        length = np.asarray(length)
        features = np.concatenate([sentence_emb, previous_emb, next_emb, section_emb], axis=1)
        features = np.column_stack([features, length])  # np.append(features,length,axis=1)
        predictions = loaded_model.predict_proba(features)

        no_sen_60 = 0
        no_sen_70 = 0
        no_sen_80 = 0
        no_sen_90 = 0
        for index, pred in enumerate(predictions):

            if pred[1] > 0.6:
                summary_text_60 += check_me.iloc[index, 0] + " "
                no_sen_60 += 1
            if pred[1] > 0.7:
                summary_text_70 += check_me.iloc[index, 0] + " "
                no_sen_70 += 1
            if pred[1] > 0.8:
                summary_text_80 += check_me.iloc[index, 0] + " "
                no_sen_80 += 1
            if pred[1] > 0.9:
                summary_text_90 += check_me.iloc[index, 0] + " "
                no_sen_90 += 1

        sentences_of_summaries_60.append(no_sen_60)
        summaries_60[paper] = summary_text_60

        sentences_of_summaries_70.append(no_sen_70)
        summaries_70[paper] = summary_text_70

        sentences_of_summaries_80.append(no_sen_80)
        summaries_80[paper] = summary_text_80

        sentences_of_summaries_90.append(no_sen_90)
        summaries_90[paper] = summary_text_90

        words_of_summaries_60.append(len(word_tokenize(summary_text_60)))

        words_of_summaries_70.append(len(word_tokenize(summary_text_70)))

        words_of_summaries_80.append(len(word_tokenize(summary_text_80)))

        words_of_summaries_90.append(len(word_tokenize(summary_text_90)))

    print("Number of words in summary using Logistic Regression as classifier and threshold equals to 0.60:",
          np.mean(words_of_summaries_60))
    print("Number of sentences in summary using Logistic Regression as classifier and threshold equals to 0.60:",
          np.mean(sentences_of_summaries_60))
    print("------------------------------------------------------")
    print()
    print("Number of words in summary using Logistic Regression as classifier and threshold equals to 0.70:",
          np.mean(words_of_summaries_70))
    print("Number of sentences in summary using Logistic Regression as classifier and threshold equals to 0.70:",
          np.mean(sentences_of_summaries_70))
    print("------------------------------------------------------")
    print()
    print("Number of words in summary using Logistic Regression as classifier and threshold equals to 0.80:",
          np.mean(words_of_summaries_80))
    print("Number of sentences in summary using Logistic Regression as classifier and threshold equals to 0.80:",
          np.mean(sentences_of_summaries_80))
    print("------------------------------------------------------")
    print()
    print("Number of words in summary using Logistic Regression as classifier and threshold equals to 0.90:",
          np.mean(words_of_summaries_90))
    print("Number of sentences in summary using Logistic Regression as classifier and threshold equals to 0.90:",
          np.mean(sentences_of_summaries_90))

    return (summaries_60,summaries_70,summaries_80,summaries_90)


def third_approach(embeddings_file):

    bert_emb = get_embeddings(embeddings_file)
    #loaded_model = joblib.load('fine_tune_BERT_sentence_classification.pkl')
    loaded_model = joblib.load('summarizer.pkl')

    test_set = create_test_set()
    papers = os.listdir("test_papers")
    summaries = dict()
    words_of_summaries = list()
    sentences_of_summaries = list()
    cosine = list()

    for paper in papers:
        print(paper)
        if paper.__eq__("250.tei.xml"):
            continue
        if paper.__eq__("289.tei.xml"):
            continue


        check_me = test_set[test_set['paper'].__eq__(paper)]
        length = list()
        sentence_emb = list()
        previous_emb = list()
        section_emb = list()
        next_list = list()
        sentence_proba = pd.DataFrame()
        nn_sentence_proba = pd.DataFrame()
        jj_redick = 0
        sentences = list()
        for row in check_me.iterrows():
            sentence = row[1][0].strip()
            sentences.append(sentence)
            previous = row[1][1].strip()
            nexts = row[1][2].strip()
            section = row[1][3].strip()

            if sentence in bert_emb:
                sentence_emb.append(bert_emb[sentence])
            else:
                sentence_emb.append(np.zeros(768))
                jj_redick += 1

            if previous in bert_emb:
                previous_emb.append(bert_emb[previous])
            else:
                previous_emb.append(np.zeros(768))

            if nexts in bert_emb:
                next_list.append(bert_emb[nexts])
            else:
                next_list.append(np.zeros(768))

            if section in bert_emb:
                section_emb.append(bert_emb[section])
            else:
                section_emb.append(np.zeros(768))

            length.append(row[1][4])

        sentence_proba['sentence'] = sentences
        sentences_set = set()
        nn_sentence_proba['sentence'] = sentences

        summary_text = ""
        sentence_emb1 = np.asarray(sentence_emb)
        next_emb = np.asarray(next_list)
        previous_emb = np.asarray(previous_emb)
        section_emb = np.asarray(section_emb)
        length = np.asarray(length)
        features = np.concatenate([sentence_emb, previous_emb, next_emb, section_emb], axis=1)
        features = np.column_stack([features, length])

        predictions = loaded_model.predict_proba(features)
        no_sen = 0
        log_preds = list()
        for i in predictions:
            log_preds.append(i[1])

        sentence_proba['probability'] = log_preds
        sentence_proba.sort_values(by=['probability'], inplace=True, ascending=False)

        from nltk import word_tokenize
        for row in sentence_proba.iterrows():
            sentence = row[1][0].strip()

            if (len(word_tokenize(summary_text)) < 300):
                sent_len = len(word_tokenize(sentence))
                if (len(word_tokenize(summary_text)) + sent_len) > 300:
                    continue
                max_cosine = 0.0
                if len(sentences_set) > 0:
                    for sen in sentences_set:
                        cos_sim = dot(bert_emb[sen], bert_emb[sentence]) / (norm(bert_emb[sentence]) * norm(bert_emb[sen]))
                        if max_cosine < cos_sim:
                            max_cosine = cos_sim
                    cosine.append(max_cosine)
                    if max_cosine < 0.90:
                        summary_text += sentence + " "
                        sentences_set.add(sentence)
                        no_sen += 1
                else:
                    summary_text += sentence + " "
                    sentences_set.add(sentence)
                    no_sen += 1

        sentences_of_summaries.append(no_sen)
        summaries[paper] = summary_text
        words_of_summaries.append(len(word_tokenize(summary_text)))

    print("Number of words in summary using Logistic Regression as classifier:", np.mean(words_of_summaries))
    print("Number of sentences in summary using Logistic Regression as classifier:", np.mean(sentences_of_summaries))
    return summaries

def evaluate(approach,embeddings_file):

    if approach.__eq__('1st'):
        summaries = first_approach(embeddings_file)
    elif approach.__eq__('3rd'):
        summaries = third_approach(embeddings_file)
    else:
        return
    posterss = os.listdir("test posters")
    posters = list()
    for poster in posterss:
        if poster.__eq__("250.pdf"):
            continue
        if poster.__eq__("289.pdf"):
            continue
        posters.append(poster)

    rouge1_log_reg = dict()
    reference_summaries_list = list()
    system_summaries_list = list()
    for p in posters:
        poster = parser.from_file("test posters/" + p)
        poster_text = poster['content']
        poster_text = re.sub(r'[^A-Za-z0-9]+', " ", poster_text).lstrip().rstrip().lower()
        paper = p[:-4] + ".tei.xml"
        reference_summaries_list.append(poster_text)
        system_summaries_list.append(summaries[paper])
        rouge1_log_reg[paper] = r.rouge_n(summaries[paper], poster_text, 1)

    f1_log_reg = list()
    for paper, rouge1 in rouge1_log_reg.items():
        f1_log_reg.append(rouge1['f'])


    print("average F1 using Rouge1 and Logistic regression as classifier is:", np.mean(f1_log_reg))
    smd_dataset = pd.DataFrame()
    smd_dataset['reference'] = reference_summaries_list
    smd_dataset['system'] = system_summaries_list

    if approach.__eq__('1st'):
        smd_dataset.to_csv('1st_approach_summaries.tsv',index=False,sep='\t',header=None)
    elif approach.__eq__('3rd'):
        smd_dataset.to_csv('3rd_approach_summaries.tsv',index=False,sep='\t',header=None)
    else:
        return


def evaluate_second_approach(embeddings_file):

    summaries = second_approach(embeddings_file)
    summaries_60 = summaries[0]
    summaries_70 = summaries[1]
    summaries_80 = summaries[2]
    summaries_90 = summaries[3]
    posterss = os.listdir("test posters")
    posters = list()
    reference_summaries_list_60 = list()
    system_summaries_list_60 = list()
    reference_summaries_list_70 = list()
    system_summaries_list_70 = list()
    reference_summaries_list_80 = list()
    system_summaries_list_80 = list()
    reference_summaries_list_90 = list()
    system_summaries_list_90 = list()

    for poster in posterss:
        if poster.__eq__("250.pdf"):
            continue
        if poster.__eq__("289.pdf"):
            continue
        posters.append(poster)

    rouge1_log_reg_60 = dict()
    rouge1_log_reg_70 = dict()
    rouge1_log_reg_80 = dict()
    rouge1_log_reg_90 = dict()

    for p in posters:
        poster = parser.from_file("test posters/" + p)
        poster_text = poster['content']
        poster_text = re.sub(r'[^A-Za-z0-9]+', " ", poster_text).lstrip().rstrip().lower()
        paper = p[:-4] + ".tei.xml"
        reference_summaries_list_60.append(poster_text)
        system_summaries_list_60.append(summaries_60[paper])
        reference_summaries_list_70.append(poster_text)
        system_summaries_list_70.append(summaries_70[paper])
        reference_summaries_list_80.append(poster_text)
        system_summaries_list_80.append(summaries_80[paper])
        reference_summaries_list_90.append(poster_text)
        system_summaries_list_90.append(summaries_90[paper])
        rouge1_log_reg_60[paper] = r.rouge_n(summaries_60[paper], poster_text, 1)

        rouge1_log_reg_70[paper] = r.rouge_n(summaries_70[paper], poster_text, 1)
        try:
            rouge1_log_reg_80[paper] = r.rouge_n(summaries_80[paper], poster_text, 1)
        except:
            print('error in',p)

        try:
            rouge1_log_reg_90[paper] = r.rouge_n(summaries_90[paper], poster_text, 1)
        except:
            print('error in',p)

    f1_log_reg_60 = list()
    for paper, rouge1 in rouge1_log_reg_60.items():
        f1_log_reg_60.append(rouge1['f'])

    f1_log_reg_70 = list()
    for paper, rouge1 in rouge1_log_reg_70.items():
        f1_log_reg_70.append(rouge1['f'])


    f1_log_reg_80 = list()
    for paper, rouge1 in rouge1_log_reg_80.items():
        f1_log_reg_80.append(rouge1['f'])

    f1_log_reg_90 = list()
    for paper, rouge1 in rouge1_log_reg_90.items():
        f1_log_reg_90.append(rouge1['f'])

    print("average F1 using Rouge1, Logistic regression as classifier and threshold 0.60 is:", np.mean(f1_log_reg_60))
    print("---------------------------------")
    print()
    print("average F1 using Rouge1, Logistic regression as classifier and threshold 0.70 is:", np.mean(f1_log_reg_70))
    print("---------------------------------")
    print()
    print("average F1 using Rouge1, Logistic regression as classifier and threshold 0.80 is:", np.mean(f1_log_reg_80))
    print("---------------------------------")
    print()
    print("average F1 using Rouge1, Logistic regression as classifier and threshold 0.90 is:", np.mean(f1_log_reg_90))

    smd_dataset_60 = pd.DataFrame()
    smd_dataset_60['reference'] = reference_summaries_list_60
    smd_dataset_60['system'] = system_summaries_list_60


    smd_dataset_60.to_csv('2nd_approach0.6_summaries.tsv', index=False, sep='\t', header=None)

    smd_dataset_70 = pd.DataFrame()
    smd_dataset_70['reference'] = reference_summaries_list_70
    smd_dataset_70['system'] = system_summaries_list_70

    smd_dataset_70.to_csv('2nd_approach0.7_summaries.tsv', index=False, sep='\t', header=None)

    smd_dataset_80 = pd.DataFrame()
    smd_dataset_80['reference'] = reference_summaries_list_80
    smd_dataset_80['system'] = system_summaries_list_80

    smd_dataset_80.to_csv('2nd_approach0.8_summaries.tsv', index=False, sep='\t', header=None)

    smd_dataset_90 = pd.DataFrame()
    smd_dataset_90['reference'] = reference_summaries_list_90
    smd_dataset_90['system'] = system_summaries_list_90

    smd_dataset_90.to_csv('2nd_approach0.9_summaries.tsv', index=False, sep='\t', header=None)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sentences", required=True, help="test sentences list")
    ap.add_argument("-o", "--output", required=True, help="test output")
    ap.add_argument("-v", "--vocab", required=True, help="path to vocab.txt")
    ap.add_argument("-c", "--config", required=True, help="path to bert_config.json")
    ap.add_argument("-m", "--model", required=True, help="path to fine-tuned model")

    args = vars(ap.parse_args())
    #extract_embeddings('C:/Users/user/PycharmProjects/bert/test_sentences_list.txt','test_output_layer_-31.json',
     #                  'D:/cased_L-12_H-768_A-12/vocab.txt','D:/cased_L-12_H-768_A-12/bert_config.json'
      #                 ,'C:/Users/user/PycharmProjects/bert/my_dataset_output/model.ckpt-3000')

    extract_embeddings(args['sentences'],args['output'],args['vocab'],args['config'],args['model'])
    evaluate('1st',args['output'])
    evaluate_second_approach(args['output'])
    evaluate('3rd',args['output'])

