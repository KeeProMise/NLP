import torch
import DataSet
import BiLSTM_CRF
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
embeding_dim = 50
hidden_dim = 256

rawfile = ".\data\\train_corpus.txt"
lablefile = ".\data\\train_label.txt"
START_TAG = "<START>"
STOP_TAG = "<STOP>"

inputs = DataSet.PreparaData(rawfile)
tags = DataSet.PreparaData(lablefile)

rawvocab = DataSet.vocab(inputs)
tagvocab = DataSet.vocab(tags)

tagvocab[0][START_TAG] = len(tagvocab[0])
tagvocab[0][STOP_TAG] = len(tagvocab[0])
rawvocab[0]["<unk>"] = len(rawvocab[0])


DataSet.save_vocab(rawvocab,".\\vocab","train_rawvocab")
DataSet.save_vocab(tagvocab,".\\vocab","train_tagvocab")

trainset = DataSet.trainDataSet(inputs,tags,rawvocab,tagvocab,device)

# model = BiLSTM_CRF.BiLSTM_CRF(len(rawvocab[0]),tagvocab[0],embeding_dim,hidden_dim)
model = torch.load(".\model\BiLSTM-CRF2")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(trainset,epoch,print_every):
    start = time.time()
    iters = len(trainset)
    iter = 1
    print(iters)
    print_loss_total = 0
    for i in range(epoch):
        epoch_loss = 0
        batch_loss = 0

        for input,tag in trainset:
            model.zero_grad()
            loss = model.neg_log_likelihood(input, tag)
            epoch_loss+=loss.item()
            batch_loss+=loss.item()
            print_loss_total += loss
            iter+=1
            loss.backward()
            optimizer.step()
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / (iters*epoch)),
                                             iter, iter / (iters*epoch) * 100, print_loss_avg))
    torch.save(model,".\model\BiLSTM-CRF3")

def main():
    train(trainset, 2, 50)


if __name__ == '__main__':
    main()

