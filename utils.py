#把word2index和index2word写入pickle文件中保存
import json
import torch
import pickle
def save_word2index(word2index, path = 'datas/word2index.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(word2index, f)
def save_index2word(index2word, path='datas/index2word.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(index2word, f)
def load_word2index(path = 'datas/word2index.pkl'):
    with open(path, 'rb') as f:
        word2index = pickle.load(f)
    return word2index
def load_index2word(path = 'datas/index2word.pkl'):
    with open(path, 'rb') as f:
        index2word = pickle.load(f)
    return index2word
def get_index2label(path = 'datas/indexlabel.json'):
    with open(path, "r", encoding="utf-8") as f:
        index2index = json.load(f)
        label2index = {value: int(key) for key, value in index2index.items()}
        index2label = {value: key for key, value in label2index.items()}
    return index2label
def get_wv(path = 'datas/wv.json'):
    with open(path, "r", encoding="utf-8") as f:
        wv = json.load(f)
    return wv
def cross_entropy(preds : torch.Tensor, targets : torch.Tensor):
    preds_softmax = torch.softmax(preds, dim = -1)
    preds_log_softmax = torch.log(preds_softmax)
    return -torch.mean(torch.sum(preds_log_softmax * targets, dim = -1))