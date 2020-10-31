import torch
import numpy as np
import torch.utils.data as Data




START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
#Read the specified file,return a sentence lists
def PreparaData(filePath):
    lines = open(filePath, encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.strip().split(' ')] for l in lines]
    return pairs

#输入句子列表，构建2个字典分别是(单词到数字，和数字到单词)字典
def vocab(pairs):
    vocab = []
    s2i = {}
    i2s = {}
    vocab.append(s2i)
    vocab.append(i2s)
    i = 0
    for pair in pairs:
        for word in pair:
            if word not in s2i:
                s2i[word] = i
                i2s[i] = word
                i += 1
    return vocab

#保存字典到文件系统
def save_vocab(vocab,savePath,filename):
    np.save("%s\%s"% (savePath, filename), vocab)

#从文件系统加载之前字典
def load_vocab(loadPath,filename):
    return np.load("%s\%s.npy"% (loadPath, filename),allow_pickle=True)


#将句子列表转换成数字串列表
def word2index(parts,vocab):
    s2i = vocab[0]
    return [[int(s2i[word]) for word in part]for part in parts]

#将数字串列表转成成句子列表
def index2word(indexs,vocab):
    i2s = vocab[1]
    return [[i2s[i] for i in index] for index in indexs]

#构建训练数据列表，列表数据格[(input1,tag1),(input2,tag2),...]
def trainDataSet(rawParts,lableParts,rawVocab,LableVocab,device =device ):
    rawParts_in = word2index(rawParts,rawVocab)
    labelParts_in = word2index(lableParts,LableVocab)
    transet = []
    for i in range(len(rawParts_in)):
        rawpart = torch.tensor(rawParts_in[i],dtype=torch.long,device=device)
        lablepart = torch.tensor(labelParts_in[i],dtype=torch.long,device=device)
        transet.append((rawpart,lablepart))
    return transet

def test_unk(train_rawvocab,pairs):
    sum = 0
    s2i = train_rawvocab[0]
    for pair in pairs:
        for word in pair:
            if word not in s2i:
                sum+=1
    return sum

def word2unk(word,trainvocab):
    if word not in trainvocab[0]:
        return "<unk>"
    else:
        return word

def testDataSet(rawPart,trainvocab,device=device):
    rawPart2trainwordvector = [[word2unk(word,trainvocab) for word in part]for part in rawPart]
    rawParts_in = word2index(rawPart2trainwordvector,trainvocab)
    testset = []
    for i in range(len(rawParts_in)):
        rawpart = torch.tensor(rawParts_in[i], dtype=torch.long, device=device)
        testset.append(rawpart)
    return testset

def test_trainDataSet(testrawPart,trainvocab,test_label_parts,train_label_vocab):
    rawPart2trainwordvector = [[word2unk(word, trainvocab) for word in part] for part in testrawPart]
    rawParts_in = word2index(rawPart2trainwordvector, trainvocab)
    labelParts_in = word2index(test_label_parts, train_label_vocab)
    testset = []
    for i in range(len(rawParts_in)):
        rawpart = torch.tensor(rawParts_in[i], dtype=torch.long, device=device)
        lablepart = torch.tensor(labelParts_in[i], dtype=torch.long, device=device)
        testset.append((rawpart, lablepart))
    return testset