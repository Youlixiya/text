import torch
import pandas as pd
from torch import nn
import pytorch_lightning as pl
from argparse import Namespace
from datasets import SkipgramDataset
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torch.optim import Adam, AdamW
from utils import load_index2word, load_word2index, get_wv
class Skipgram(pl.LightningModule):
    def __init__(self, args : Namespace):
        super(Skipgram, self).__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.n_gram = args.n_gram
        self.datas_df = pd.read_csv(args.datas_csv_file_path)
        self.word2index, self.index2word = load_word2index(), load_index2word()
        self.words_num = len(self.word2index)
        self.vocab = nn.Embedding(self.words_num, args.embedding_dim)
        self.fc = nn.Linear(args.embedding_dim, self.words_num)
        self.train_loss = MeanMetric()
        self.loss_fn = nn.CrossEntropyLoss()
    def setup(self, stage: str):
        self.train_dataset = SkipgramDataset(self.n_gram, self.datas_df, self.args.wv_path, self.word2index, self.index2word)
    def configure_optimizers(self):
        self.optimizer = eval(self.args.optimizer)(self.parameters(), lr=self.args.lr)
        return self.optimizer
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.args.workers,
                          batch_size=self.args.batch_size,
                          pin_memory=True)
    def training_step(self, batch, batch_nb):
        datas, targets = batch
        b = datas.shape[0]
        preds = self.fc(self.vocab(datas).reshape(b, self.args.embedding_dim))
        loss = self.loss_fn(preds, targets.reshape(b))
        self.train_loss.update(loss)
        self.log('train_loss', self.train_loss(loss), prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.args.use_wandb else False)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        self.log( "men", mem, prog_bar=True, on_step=True, on_epoch=False,
            logger=True if self.args.use_wandb else False,
        )
        return loss
    def on_train_epoch_end(self) -> None:
        mean_loss = self.train_loss.compute()
        self.log('train_mean_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.train_loss.reset()