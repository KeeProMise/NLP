import torch
import DataSet
import BiLSTM_CRF
import numpy as np


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

rawfile = ".\data\\train_corpus.txt"
lablefile = ".\data\\train_label.txt"

inputs = DataSet.PreparaData(rawfile)
tags = DataSet.PreparaData(lablefile)

rawvocab = DataSet.load_vocab(".\\vocab","train_rawvocab")
tagvocab = DataSet.load_vocab(".\\vocab","train_tagvocab")

testset = DataSet.trainDataSet(inputs,tags,rawvocab,tagvocab,device)

model = torch.load(".\model\BiLSTM-CRF1")
model.to(device)

import time


def test(testset):
    iters = len(testset)
    start = time.time()
    tag_list = []
    now = 0
    with torch.no_grad():
        for iter, _ in testset:

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
def main():
    tag_list=test(testset)
    label_list =tagi2label(tag_list,tagvocab)
    write2fileSystem(".\data\mytrain_label.txt", label_list)


if __name__ == '__main__':
    main()