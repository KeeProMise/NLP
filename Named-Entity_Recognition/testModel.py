import torch
import DataSet
import BiLSTM_CRF
import numpy as np
import sklearn.metrics as score
import evaluation

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

rawfile = ".\data\\test_corpus.txt"

inputs = DataSet.PreparaData(rawfile)

rawvocab = DataSet.load_vocab(".\\vocab","train_rawvocab")
tagvocab = DataSet.load_vocab(".\\vocab","train_tagvocab")

testset = DataSet.testDataSet(inputs,rawvocab,device)

model = torch.load(".\model\BiLSTM-CRF5")
model.to(device)

import time


def test(testset):
    iters = len(testset)
    start = time.time()
    tag_list = []
    y_true = []

    now = 0
    with torch.no_grad():
        for iter in testset:
            score,perdict = model(iter)
            tag_list.append(perdict)
            if now%500 == 0:
                nowtime = time.time()
                print(nowtime-start)

    return(tag_list)

def tagi2label(tag_list,tagvocab):
    i2s = tagvocab[1]
    return [[i2s[tag] for tag in tags]for tags in tag_list ]

def write2fileSystem(filepath,labellist):
    file_handle = open(filepath, mode='w', encoding='utf-8')
    file_list = []
    for list in labellist:
        str = ""
        for label in list:
            str = str+label+" "
        str = str+"\n"
        file_list.append(str)
    file_handle.writelines(file_list)

def evaluate(predict_filepath,true_filepath):
    patten = [("B-PER", "I-PER"), ("B-LOC", "I-LOC"), ("B-ORG", "I-ORG")]
    predict_label = DataSet.PreparaData(predict_filepath)
    true_label = DataSet.PreparaData(true_filepath)
    evaluation.NER_score(patten, true_label, predict_label)


def main():
    # tag_list=test(testset)
    # label_list =tagi2label(tag_list,tagvocab)
    # write2fileSystem(".\data\mytest_label1.txt", label_list)
    evaluate(".\data\mytest_label1.txt", ".\data\\test_label.txt")

if __name__ == '__main__':
    main()