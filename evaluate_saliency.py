import os
import hydra
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

from src.datasets.dataset import SaliencyDataset
from src.models.classifier import ClassifierModule
from src.saliency_metrics import SaliencyMetrics, Insertion, Deletion
from src.utils import load_dataset, load_saliecy_method


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    loggers = None

    # instantiate the model and load the weights
    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=cfg.train.finetune,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    model.eval()

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        # model.load_state_dict(torch.load(model_path)['state_dict'])
        model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    test = SaliencyDataset(test)
    dataloader = torch.utils.data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    insertion_metric = Insertion(model, n_pixels=50)
    deletion_metric = Deletion(model, n_pixels=50)

    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    # load saliency method
    saliency_method = load_saliecy_method(cfg.saliency.method, model, device=cfg.train.device)

    for j, (images, labels) in enumerate(dataloader):

        images = images.to(cfg.train.device)
        model = model.to(cfg.train.device)

        saliency = saliency_method.generate_saliency(input_images=images, target_layer=target_layer).to(
            cfg.train.device)

        for i in range(images.shape[0]):
            image = images[i]
            image = image.to(cfg.train.device)
            saliency_map = saliency[i]
            saliency_map.to(cfg.train.device)
            label = labels[i]
            label = label.to(cfg.train.device)

            image_to_mask = image.clone()  # Start with the original image

            auc_ins_score = insertion_metric(image_to_mask, saliency_map, label, start_with_blurred=True)
            auc_del_score = deletion_metric(image_to_mask, saliency_map, label, start_with_blurred=False)

            print(f'AUC Insertion Score: {auc_ins_score}')
            print(f'AUC Deletion Score: {auc_del_score}')

        if j == 0:
            break


if __name__ == "__main__":
    main()
