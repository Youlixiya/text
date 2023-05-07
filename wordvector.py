import numpy as np
import torch
from models import Skipgram
from configs import skip_gram_config
from utils import load_word2index, load_index2word
from typing import Union, List
class WordVector:
    def __init__(self, ckpt_path, word2index = None, index2word = None, args = skip_gram_config):
        self.word2index, self.index2word =word2index if word2index else load_word2index(), index2word if index2word else load_index2word()
        model = Skipgram.load_from_checkpoint(checkpoint_path = ckpt_path, args = args, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = args.embedding_dim
        self.vocab = model.vocab
    def get_wordvector(self, index : Union[str, int], type : str = 'np'):
        if isinstance(index, str):
            vector = self.vocab(torch.LongTensor([self.word2index[index]]).to(self.device)).reshape(self.embedding_dim)
        elif isinstance(index, int):
            vector = self.vocab(torch.LongTensor([index]).to(self.device)).reshape(self.embedding_dim)
        else:
            print('Please re-enter, input must be a string or integer!')
        if type == 'np':
            return vector.detach().cpu().numpy()
        elif type == 'pt':
            return vector.cpu()
        else:
            print('Please enter the correct type, such as pt or np!')
    def get_batch_wordvector(self, index_list : List, type : str = 'np'):
        res = []
        for i in index_list:
            res.append(self.get_wordvector(i, type))
        if type == 'np':
            return np.array(res)
        elif type == 'pt':
            return torch.tensor(res)
        else:
            print('Please enter the correct type, such as pt or np!')

if __name__ == '__main__':
    model = Skipgram.load_from_checkpoint(checkpoint_path='skipgram/ckpts/epoch=99-train_mean_loss=4.758869171142578.ckpt', args=skip_gram_config)
    print(model.vocab)