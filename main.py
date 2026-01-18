import os, sys, hydra, multiprocessing, wandb, warnings
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from depthcharge.tokenizers import PeptideTokenizer
from do_novo.lightning_module import DenovoLightningModule
from do_novo.data_module import DenovoDataModule

warnings.filterwarnings("ignore", category=UserWarning)

work_dir = "/your/path/Do-Novo"
sys.path.append(work_dir)
os.chdir(work_dir)

CONFIG_NAME = "train_oc_ump.yaml"

def get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def get_per_rank_workers(cfg):
    if cfg.data_module.dataloader.num_workers == 'auto':
        return multiprocessing.cpu_count() // (len(cfg.trainer.train.devices)*2)
    else:
        return cfg.data_module.dataloader.num_workers


def build_callbacks(cfg):
    if cfg.logger:
        callbacks = [
            RichProgressBar(), 
            LearningRateMonitor(**cfg.callbacks.lr_monitor),
            ModelCheckpoint(**cfg.callbacks.checkpoint)
            # EarlyStopping(**cfg.callbacks.early_stopping),
        ]
        return callbacks
    else:
        return [RichProgressBar()]


def build_logger(cfg):
    if not cfg.logger:
        return False

    if get_rank() != 0:
        return False

    return WandbLogger(**cfg.wandb)


def _login_wandb_if_needed(cfg):
    if os.environ.get("WANDB_MODE", "").lower() == "offline":
        return
    key = getattr(cfg, "key", False)
    os.environ["WANDB_API_KEY"] = str(key)
    try:
        wandb.login()
    except Exception as e:
        print(f"[WARN] wandb.login() failed: {e}")


@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def main(config):
    
    seed_everything(config.seed)
    config.data_module.dataloader.num_workers = get_per_rank_workers(config)

    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True, start_token='BOS', stop_token='EOS')
    data_module = DenovoDataModule(config['data_module']['dataset'], config['data_module']['dataloader'], tokenizer)
    if config.trainer.ckpt_path:
        lightning_module = DenovoLightningModule.load_from_checkpoint(tokenizer=tokenizer, checkpoint_path=config.trainer.ckpt_path, strict=False, map_location='cpu', weights_only=False, **config.model)
    else:
        lightning_module = DenovoLightningModule(tokenizer=tokenizer, **config.model)

    callbacks = build_callbacks(config)
    logger = build_logger(config)

    if config.mode == "train":
        resume = config.trainer.ckpt_path if config.model.train_phase != 'sampler' else None
        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            precision="16-mixed",
            **config.trainer.train
        )
        trainer.fit(lightning_module, datamodule=data_module, ckpt_path=resume)

    elif config.mode == "test":
        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            precision="16-mixed",
            **config.trainer.test
        )
        trainer.test(lightning_module, datamodule=data_module)

    elif config.mode == "val":
        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            precision="16-mixed",
            **config.trainer.train,
        )
        trainer.validate(lightning_module, datamodule=data_module)

    elif config.mode == "predict":
        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            precision="16-mixed",
            **config.trainer.predict,
        )
        trainer.predict(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
