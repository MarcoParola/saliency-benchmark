import hydra
import torch
import pytorch_lightning as pl

from src.utils import load_model, load_dataset


@hydra.main(config_path='config', config_name='config')
def main(cfg):

    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)

    train, val, test = load_dataset(cfg.dataset.name) 
    train_loader = torch.utils.data.DataLoader(
        train, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)


    model = load_model(cfg.dataset.name, cfg.model)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        logger=None, # TODO add logger
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=True
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()