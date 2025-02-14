import os
from typing import Callable, Dict, List, Tuple, Union

import numpy
import torch
import torchvision
from matplotlib import pyplot as plt
import lrp

import lrp.plot
from lrp import image, rules
from lrp.core import LRP
from lrp.filter import LayerFilter
from lrp.rules import LrpEpsilonRule, LrpGammaRule, LrpZBoxRule, LrpZeroRule
from lrp.zennit.types import AvgPool, Linear
from torch import nn

from src.datasets.classification import load_classification_dataset
from src.models.classifier import ClassifierModule
from src.utils import *
import torch.utils.data as data
import hydra

class lrp_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model.model[0]
        self.device = device
        self.kwargs = kwargs

        self.name_map = None
        self.name_map: List[Tuple[List[str], rules.LrpRule,
        Dict[str, Union[torch.Tensor, float]]]]

    def generate_saliency(self, input_images, batch, resize, target_class=None, target_layer=None):
        # Low and high parameters for zB-rule
        batch_size = batch
        shape: Tuple[int] = (batch_size, 3, resize, resize)

        low: torch.Tensor = lrp.norm.ImageNetNorm.normalize(torch.zeros(*shape))
        high: torch.Tensor = lrp.norm.ImageNetNorm.normalize(torch.ones(*shape))

        # Init layer filter
        target_types: Tuple[type] = (Linear, AvgPool)
        filter_by_layer_index_type: LayerFilter = LayerFilter(model=self.model,
                                                              target_types=target_types)

        # LRP Composite from Montavon's lrp-tutorial
        self.name_map = [
            (filter_by_layer_index_type(lambda n: n == 0), LrpZBoxRule, {'low': low, 'high': high}),
            (filter_by_layer_index_type(lambda n: 1 <= n <= 16), LrpGammaRule, {'gamma': 0.25}),
            (filter_by_layer_index_type(lambda n: 17 <= n <= 30), LrpEpsilonRule, {'epsilon': 0.25}),
            (filter_by_layer_index_type(lambda n: 31 <= n), LrpZeroRule, {}),
        ]
        # Init LRP
        lrp_instance = LRP(self.model)

        # Prepare model layers for LRP
        #print(target_layers)
        lrp_instance.convert_layers(self.name_map)

        # Initialize the tensor to store saliency maps for the batch
        batch_saliencies = []

        # Generate the saliency maps for each image in the batch
        for image in input_images:

            # Compute relevance attributions
            saliency_map: torch.Tensor = lrp_instance.relevance(image.unsqueeze(0).to(self.device))

            # Plot saliency
            #lrp_instance.heatmap(saliency_map, width=2, height=2)

            # Append the saliency map to the batch_saliencies list
            batch_saliencies.append(saliency_map.sum(dim=1))
            #batch_saliencies.append(saliency_map)

        # Stack the saliency maps along the batch dimension
        saliency_maps = torch.stack(batch_saliencies)

        return saliency_maps
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    print("LRP")
    print(cfg.model)
    print(cfg.dataset.name)
    print(cfg.checkpoint)
    torch.cuda.empty_cache()
    # Load the model and data
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

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataloader = data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    if save_images:
        # Create directory to save saliency maps
        finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
        output_dir_images = os.path.join(cfg.currentDir, 'saliency_output/lrp_saliency_maps_images', finetune + cfg.model + cfg.dataset.name)
        output_dir_tensors = os.path.join(cfg.currentDir, 'saliency_output/lrp_saliency_maps_tensors', finetune + cfg.model + cfg.dataset.name)
        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_tensors, exist_ok=True)

    # Initialize the Saliency method
    method = lrp_interface(model, device=cfg.train.device, reshape_transform=False)

    image_count = 0

    for images, labels in dataloader:
        # if image_count >= 2:
        #      break

        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(input_images=images, batch=cfg.train.batch_size,resize=cfg.dataset.resize)

        for i in range(images.size(0)): #i-th images of the batch
            # if image_count >= 2:
            #      break
            print(image_count)

            image = images[i]
            image = image.cpu()
            saliency = saliency_maps[i].cpu()
            predicted_class = preds[i]
            true_class = labels[i]

            # Create figure
            fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
            ax[0].imshow(image.squeeze().permute(1, 2, 0))
            ax[0].axis('off')
            ax[1].imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
            ax[1].axis('off')
            ax[1].set_title(f'Pred: {predicted_class}\nTrue: {true_class}')

            if save_images:
                if image_count<=200:
                    output_path = os.path.join(output_dir_images, f'saliency_map_{image_count}.png')
                    plt.savefig(output_path)
                save_saliency_map(os.path.join(output_dir_tensors, f'saliency_map_{image_count}.pth'), saliency)
            else:
                plt.show()

            plt.close(fig)

            image_count += 1

if __name__ == '__main__':
    main()
