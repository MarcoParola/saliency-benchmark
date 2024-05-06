import hydra
import torch
import pytorch_lightning as pl
import os

from src.utils import load_dataset, get_early_stopping, get_save_model_callback
from src.models.classifier import ClassifierModule
from src.log import get_loggers



@hydra.main(config_path='config', config_name='config')
def main(cfg):

    # Set seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)

    # callback
    callbacks = list()
    callbacks.append(get_early_stopping(cfg.train.patience))
    model_save_dir = os.path.join(cfg.currentDir, cfg.checkpoint, cfg.model + cfg.dataset.name )
    callbacks.append(get_save_model_callback(model_save_dir))


    # loggers
    loggers = get_loggers(cfg)

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    train_loader = torch.utils.data.DataLoader(train, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.num_workers)
    val_loader = torch.utils.data.DataLoader(val, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)
    test_loader = torch.utils.data.DataLoader(test, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)


    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes, 
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        logger=loggers,
        log_every_n_steps=cfg.train.log_every_n_steps,
        #deterministic=True
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)




if __name__ == '__main__':
    main()