import json
import os

import cv2
import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset
from src.models.classifier import ClassifierModule
from src.models.detector.grounded_sam2 import GroundedSam2
from src.utils import save_mask, load_mask, retrieve_concepts, save_list, save_images_with_mask_for_all_concepts, \
    load_saliency_method

if GlobalHydra().is_initialized():
    GlobalHydra().clear()


# Script designed to generate an plot with [image, image + saliency, image + concept, image + saliency + concept] for a single image

@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    caption = retrieve_concepts(cfg.dataset.name)

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

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataset = test
    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    # load saliency method
    saliency_method = load_saliency_method(cfg.saliency.method, model, device=cfg.train.device)

    # Put here the number of image desired
    image_id_desired = 5

    if cfg.modelSam == 'Florence2':
        model_sam = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
    elif cfg.modelSam == 'GroundingDino':
        model_sam = GroundedSam2(caption, 'Grounding DINO')

    for idx in range(image_id_desired, image_id_desired + 1):
        print("IMG " + str(idx))

        # Get the ground truth bounding boxes and labels
        image, ground_truth_labels = dataset.__getitem__(idx)

        # Predict using the model
        boxes, masks, classes = model_sam.mask_generation_with_all_concepts(ToPILImage()(image),
                                                                            cfg.dataset.resize)

        image = image.unsqueeze(0).to(device)

        # Get model predictions
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

        saliency = saliency_method.generate_saliency(input_images=image, target_layer=target_layer).to(
            cfg.train.device)

        image = image.cpu()
        saliency = saliency.cpu()
        i = 0
        mask_head_paws = np.expand_dims(np.full((cfg.dataset.resize, cfg.dataset.resize), False, dtype=bool), axis=0)
        for mask, concept in zip(masks, classes):

            if model_sam.ontology.classes()[concept] == 'Head' or model_sam.ontology.classes()[concept] == 'Paws':
                mask_head_paws = np.logical_or(mask_head_paws, mask)

            # Create figure
            fig, ax = plt.subplots(1, 4, figsize=(10, 5))
            ax[0].imshow(image.squeeze().permute(1, 2, 0))
            ax[0].axis('off')
            ax[1].imshow(image.squeeze().permute(1, 2, 0))
            ax[1].imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
            ax[1].axis('off')
            ax[1].set_title(f'label: {dataset.classes[preds]}')
            ax[2].imshow(image.squeeze().permute(1, 2, 0))
            ax[2].imshow(mask, cmap='jet', alpha=0.4)
            ax[2].axis('off')
            ax[2].set_title(f'concept: {model_sam.ontology.classes()[concept]}')
            ax[3].imshow(image.squeeze().permute(1, 2, 0))
            ax[3].imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
            ax[3].imshow(mask, cmap='jet', alpha=0.4)
            ax[3].axis('off')
            ax[3].set_title(f'label and concept')
            plt.savefig(
                os.path.join(os.path.abspath('output'), f'plot_fusion_concept_{cfg.saliency.method}_{i}.jpg'))
            i += 1

        output_dir = os.path.abspath('output')
        os.makedirs(output_dir, exist_ok=True)

        # Original image
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'1original_{cfg.saliency.method}.pdf'), bbox_inches='tight')
        plt.close()

        # Saliency map
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'1saliency_{cfg.saliency.method}.pdf'), bbox_inches='tight')
        plt.close()

        # Concept mask
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.imshow(mask_head_paws.squeeze(), cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'concept_{cfg.saliency.method}.pdf'), bbox_inches='tight')
        plt.close()

        # Combined saliency & concept mask
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
        plt.imshow(mask_head_paws.squeeze(), cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'fusion_{cfg.saliency.method}.pdf'), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
