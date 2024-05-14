import torch
import hydra
import torchvision
import os

from src.models.classifier import ClassifierModule

@hydra.main(config_path='../config', config_name='config')
def main(cfg):

    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes, 
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )
    model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
    model.load_state_dict(torch.load(model_path)['state_dict'])

    for name, module in model.named_modules():
        print(name)




if __name__ == '__main__':
    main()