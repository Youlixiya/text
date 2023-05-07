from configs import skip_gram_config
from models import Skipgram
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
def main():
    args = skip_gram_config
    seed_everything(args.seed)
    if args.use_wandb:
        if args.wandb_id:
            wandb_logger = WandbLogger(
                    project=args.project_name,
                    log_model=True,
                    id=args.wandb_id,
                    resume="must")
        else:
             wandb_logger = WandbLogger(
                    project=args.project_name,
                    log_model=True)
    else:
        wandb_logger=None
    model = Skipgram(args)
    checkpoint_callback = ModelCheckpoint(dirpath=f'{args.exp_name}/ckpts',
                                          filename='{epoch}-{train_mean_loss}',
                                          monitor='train_mean_loss',
                                          mode='min',
                                          save_top_k=3,)


    trainer = Trainer(accelerator='auto',
                      max_epochs=args.num_epochs,
                      callbacks=checkpoint_callback,
                      logger=wandb_logger,
                      devices=args.devices,
                      precision=args.precision,
                      log_every_n_steps=1,
                      num_sanity_val_steps=0,
                      benchmark=True)
    if args.ckpt_path:
        trainer.fit(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)
if __name__ == '__main__':
    main()