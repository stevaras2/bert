import json
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.utils.vis_utils import plot_model
from sklearn.externals import joblib


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_embeddings(sentences_list,layer_json):
    '''

    :param sentences_list: the path o the sentences.txt
    :param layer_json: the path of the json file that contains the embeddings of the sentences
    :return: Dictionary with key each sentence of the sentences_list and as value the embedding
    '''
    sentences = dict()#dict with key the index of each line of the sentences_list.txt and as value the sentence
    embeddings = dict()##dict with key the index of each sentence and as value the its embedding
    sentence_emb = dict()#key:sentence,value:its embedding

    with open(sentences_list,'r') as file:
        for index,line in enumerate(file):
            sentences[index] = line.strip()

    with open(layer_json, 'r',encoding='utf-8') as f:
        for line in f:
            embeddings[json.loads(line)['linex_index']] = np.asarray(json.loads(line)['features'])



    for key,value in sentences.items():
        sentence_emb[value] = embeddings[key]


    return sentence_emb

def train_classifier(sentences_list,layer_json,dataset_csv,filename):
    '''

    :param sentences_list: the path o the sentences.txt
    :param layer_json: the path of the json file that contains the embeddings of the sentences
    :param dataset_csv: the path of the dataset
    :param filename: The path of the pickle file that the model will be stored
    :return:
    '''

    dataset = pd.read_csv(dataset_csv)
    bert_dict = get_embeddings(sentences_list,layer_json)


    length = list()
    sentence_emb = list()
    previous_emb = list()
    next_list = list()
    section_list = list()
    label = list()
    # print(np.zeros(768).shape)
    errors = 0
    for row in dataset.iterrows():
        sentence = row[1][0].strip()
        previous = row[1][1].strip()
        nexts = row[1][2].strip()
        section = row[1][3].strip()

        if sentence in bert_dict:
            sentence_emb.append(bert_dict[sentence])
        else:
            sentence_emb.append(np.zeros(768))
            print(sentence)
            errors += 1

        if previous in bert_dict:
            previous_emb.append(bert_dict[previous])
        else:
            previous_emb.append(np.zeros(768))

        if nexts in bert_dict:
            next_list.append(bert_dict[nexts])
        else:
            next_list.append(np.zeros(768))

        if section in bert_dict:
            section_list.append(bert_dict[section])
        else:
            section_list.append(np.zeros(768))

        length.append(row[1][4])
        label.append(row[1][5])

    sentence_emb = np.asarray(sentence_emb)
    print(sentence_emb.shape)
    next_emb = np.asarray(next_list)
    print(next_emb.shape)
    previous_emb = np.asarray(previous_emb)
    print(previous_emb.shape)
    section_emb = np.asarray(section_list)
    print(sentence_emb.shape)
    length = np.asarray(length)
    print(length.shape)
    label = np.asarray(label)
    print(errors)
    features = np.concatenate([sentence_emb, previous_emb, next_emb,section_emb], axis=1)
    features = np.column_stack([features, length])  # np.append(features,length,axis=1)
    print(features.shape)

    X_train, X_val, y_train, y_val = train_test_split(features, label, test_size=0.33, random_state=42)

    log = LogisticRegression(random_state=0, solver='newton-cg', max_iter=1000, C=0.1)
    log.fit(X_train, y_train)

    #save the model
    _ = joblib.dump(log, filename, compress=9)

    predictions = log.predict(X_val)
    print("###########################################")
    print("Results using embeddings from the",layer_json,"file")
    print(classification_report(y_val, predictions))
    print("F1 score using Logistic Regression:",f1_score(y_val, predictions))
    print("###########################################")


    #train a DNN
    f1_results = list()
    for i in range(3):
        model = Sequential()
        model.add(Dense(64, activation='relu', trainable=True))
        model.add(Dense(128, activation='relu', trainable=True))
        model.add(Dropout(0.30))
        model.add(Dense(64, activation='relu', trainable=True))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu', trainable=True))
        model.add(Dropout(0.35))
        model.add(Dense(1, activation='sigmoid'))
        # compile network
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[f1])
        # fit network
        model.fit(X_train, y_train, epochs=100, batch_size=64)



        loss, f_1 = model.evaluate(X_val, y_val, verbose=1)
        print('\nTest F1: %f' % (f_1 * 100))
        f1_results.append(f_1)
        model = None

    print("###########################################")
    print("Results using embeddings from the", layer_json, "file")
    # evaluate
    print(np.mean(f1_results))
    print("###########################################")

def visualize_DNN(file_to_save):
    '''
    Save the DNN architecture to a png file. Better use the Visulize_DNN.ipynd
    :param file_to_save: the png file that the architecture of the DNN will be saved.
    :return: None
    '''

    model = Sequential()
    model.add(Dense(64, activation='relu', trainable=True))
    model.add(Dense(128, activation='relu', trainable=True))
    model.add(Dropout(0.30))
    model.add(Dense(64, activation='relu', trainable=True))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu', trainable=True))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))

    plot_model(model, to_file=file_to_save, show_shapes=True)



if __name__ == '__main__':

    #layer_1 = train_classifier('sentences_list.txt','output_layer_-1.json','train_sentences1.csv','fine_tune_BERT_sentence_classification.pkl')
    #layer_2 = train_classifier('sentences_list.txt','output_layer_-2.json','train_sentences1.csv','fine_tune_BERT_sentence_classification.pkl')
    #layer_3 = train_classifier('sentences_list.txt','output_layer_-3.json','train_sentences1.csv','fine_tune_BERT_sentence_classification.pkl')
    layer_4 = train_classifier('sentences_list.txt','output_layer_-4.json','train_sentences1.csv','fine_tune_BERT_sentence_classification.pkl')

