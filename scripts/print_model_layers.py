import torch
import hydra
import torchvision
import os

from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights, resnet34, VGG11_Weights, vgg11

from src.models.classifier import ClassifierModule


@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.mainDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    for name, module in model.named_modules():
        print(name)


if __name__ == '__main__':
    main()
