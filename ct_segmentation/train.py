import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from ct_segmentation.lightning_modules.pl_unet import UNetModel
from ct_segmentation.lightning_data_modules.ctdataset import CTDataset
import torch

@hydra.main(config_path="../config", config_name="config")
def train(cfg: DictConfig):
    print(f"Training experiment: {cfg.experiment_name}")
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    model = hydra.utils.instantiate(cfg.model)
    wandb.login()
    loggers = [hydra.utils.instantiate(logger) for logger in cfg.trainer.logger]
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=loggers,
        accelerator="gpu",
        precision=cfg.trainer.precision,
    )
    trainer.fit(model, datamodule=dm)
    wandb.finish()
    return model


if __name__ == "__main__":
    train()
    model = model_save_path = "/content/unet.pth"
    torch.save(model, model_save_path)