from argparse import Namespace
skip_gram_config = Namespace(
    num_epochs = 100,
    batch_size = 512,
    seed = 0,
    embedding_dim = 512,
    n_gram = 2,
    datas_csv_file_path = 'datas/datas_index.csv',
    wv_path = None,
    optimizer = 'AdamW',
    lr = 1e-3,
    workers = 16,
    exp_name = 'skipgram1',
    devices = 1,
    precision = '32',
    ckpt_path = None,
    use_wandb = False,
    wandb_id = None,
    project_name = 'skipgram',
)