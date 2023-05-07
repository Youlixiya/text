import torch
import pandas as pd
from utils import load_index2word, load_word2index, get_wv
from torch.utils.data import Dataset
class SkipgramDataset(Dataset):
    def __init__(self, n_gram=2, datas_df = None, wv_path = None, word2index = None, index2word = None):
        self.n_gram = n_gram
        self.wv = get_wv(wv_path) if wv_path else None
        self.datas_des, self.datas_target = datas_df.iloc[:, 0], datas_df.iloc[:, 1]
        self.word2index, self.index2word =word2index if word2index else load_word2index(), index2word if index2word else load_index2word()
        self.words_num = len(self.word2index)
        self.train_datas_list = self.get_train_datas()
        self.datas_des_wv = self.index2wv() if wv_path else None
    def get_train_datas(self):
        train_datas_list = []
        datas_des = self.datas_des.tolist()
        for datas in datas_des:
            datas = eval(datas)
            total_words = len(datas)
            for i in range(self.n_gram, total_words - self.n_gram):
                target_index_list = list(range(i - self.n_gram, i)) + list(range(i + 1, i + self.n_gram + 1))
                for j in target_index_list:
                    train_datas_list.append([datas[i], datas[j]])
        return train_datas_list
    def index2wv(self):
        datas_des_wv = []
        datas_des = self.datas_des.tolist()
        for datas in datas_des:
            datas = eval(datas)
            temp = []
            for index in datas:
                temp.append(self.wv[self.index2word[index]])
            datas_des_wv.append(temp)
    def __len__(self):
        if self.wv:
            return len(self.datas_target)
        else:
            return len(self.train_datas_list)
    def __getitem__(self, idx):
        if self.wv:
            return torch.FloatTensor(self.datas_des_wv[idx]), torch.LongTensor([self.datas_target[idx]])
        else:
            return torch.LongTensor([self.train_datas_list[idx][0]]), torch.LongTensor([self.train_datas_list[idx][1]])
if __name__ == '__main__':
    datas_df = pd.read_csv('datas/datas_index.csv')
    dataset = SkipgramDataset(datas_df=datas_df)
    print(dataset[0])