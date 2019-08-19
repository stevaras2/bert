from xml.etree import ElementTree
import os
from nltk import sent_tokenize,word_tokenize
import re
import pandas as pd

def get_text_per_section(train_papers):
    xml_files = os.listdir(train_papers)
    dict_of_sections = dict()
    for f in xml_files:
        print(f)
        file = ElementTree.parse(train_papers+"/"+f)
        #root = file.getroot()
        sections = dict()

        for div in file.getiterator(tag="{http://www.tei-c.org/ns/1.0}div"):

            sec = ""
            for head in div.iter('{http://www.tei-c.org/ns/1.0}head'):
                if head.text not in sections:
                    sections[head.text] = ""
                    sec = head.text

            text = ""
            for p in div.iter('{http://www.tei-c.org/ns/1.0}p'):
                text += p.text
            sections[sec] = text
        dict_of_sections[f] = sections

    return dict_of_sections

def create_dataset(train_papers,dataset_path):

    sections_per_text = get_text_per_section(train_papers)


    train_dataset = dict()# dictionary that will be use for the fine-tune of the BERT model in our data.
    #Key:The first three sentences of each section. Value: Each sentence of the section
    clipping_id_list = list()
    sentence_id_list = list()
    clip_sections = list()
    sentence_list = list()
    clipping_id = 0
    sentence_id = 0
    for file,sections in sections_per_text.items():

        for section,text in sections.items():

            sen_num = 0
            clipping = ""
            for sentence in sent_tokenize(text):
                clipping += sentence + " "
                sen_num += 1
                if sen_num > 2:
                    break

            clipping = re.sub(r'[^A-Za-z0-9]+'," ",clipping.strip()).lower()
            for sentence in sent_tokenize(text):

                sentence = re.sub(r'[^A-Za-z0-9]+'," ",sentence).lower()
                if len(word_tokenize(sentence)) > 2:
                    if clipping not in train_dataset:
                        train_dataset[clipping] = [sentence]
                    else:
                        train_dataset[clipping].append(sentence)

                    clipping_id_list.append(clipping_id)
                    sentence_id_list.append(sentence_id)
                    sentence_id += 1
                    clip_sections.append(clipping)
                    sentence_list.append(sentence)

        clipping_id += 1

    dataset = pd.read_csv(dataset_path)

    dataset_sentences = dict()
    labels_list = list()
    error = 0
    for row in dataset.iterrows():

        dataset_sentences[row[1][0]] = row[1][5]
        if row[1][0] not in sentence_list:
            error += 1

    er = 0
    for sentence in sentence_list:

        if sentence not in dataset_sentences:
            labels_list.append(0)
            er += 1
        else:
            labels_list.append(dataset_sentences[sentence])

    dataset_for_fine_tune = pd.DataFrame()
    dataset_for_fine_tune['Label'] = labels_list
    dataset_for_fine_tune['id1'] = clipping_id_list
    dataset_for_fine_tune['id2'] = sentence_id_list
    dataset_for_fine_tune['clipping'] = clip_sections
    dataset_for_fine_tune['sentence'] = sentence_list

    dataset_for_fine_tune.iloc[:, :].to_csv(os.path.join("PG", "full_dataset.tsv"), index=None, sep="\t")
    dataset_for_fine_tune.iloc[:8000,:].to_csv(os.path.join("PG","train.tsv"),index=None,sep="\t")
    dataset_for_fine_tune.iloc[8001:9500,:].to_csv(os.path.join("PG","dev.tsv"),index=None,sep="\t")
    dataset_for_fine_tune.iloc[9501:, :].to_csv(os.path.join("PG", "test.tsv"), index=None, sep="\t")



def create_another_dataset(dataset_path):


    dataset = pd.read_csv(dataset_path)
    textA_list = list()
    textB_list = list()
    label_list = list()
    features_id_list = list()
    sentence_id_list = list()
    id = 0
    for row in dataset.iterrows():
        textA = ""

        textA += str(row[1][0]) + "." + str(row[1][1]) + "."+ str(row[1][2]) + "." + str(row[1][3])
        textB = str(row[1][0])
        textA_list.append(textA)
        textB_list.append(textB)
        label_list.append(row[1][5])
        sentence_id_list.append(id)
        f_id = 100000 + id
        features_id_list.append(f_id)
        id += 1

    dataset_for_fine_tune = pd.DataFrame()
    dataset_for_fine_tune['Label'] = label_list
    dataset_for_fine_tune['id1'] = features_id_list
    dataset_for_fine_tune['id2'] = sentence_id_list
    dataset_for_fine_tune['features'] = textA_list
    dataset_for_fine_tune['sentence'] = textB_list


    dataset_for_fine_tune.iloc[:8000, :].to_csv(os.path.join("PG1", "train.tsv"), index=None, sep="\t")
    dataset_for_fine_tune.iloc[8001:9500, :].to_csv(os.path.join("PG1", "dev.tsv"), index=None, sep="\t")
    dataset_for_fine_tune.iloc[9501:, :].to_csv(os.path.join("PG1", "test.tsv"), index=None, sep="\t")

    return dataset_for_fine_tune

#dataset = create_another_dataset('train_sentences1.csv')
#create_dataset('train_papers','train_sentences1.csv')
create_dataset('C:/Users/user/PycharmProjects/other_dataset/train_papers_xml','C:/Users/user/PycharmProjects/other_dataset/train_sentences1.csv')