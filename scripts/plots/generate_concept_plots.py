import csv
import os

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset
from src.models.detector.grounded_sam2 import GroundedSam2
from src.utils import save_mask, load_mask, retrieve_concepts

if GlobalHydra().is_initialized():
    GlobalHydra().clear()
import math


def find_closest_factors(n):
    # Start from the square root of n
    a = math.isqrt(n)  # integer part of sqrt(n)

    # Find the closest factors
    best_a, best_b = a, math.ceil(n / a)
    min_diff = abs(best_a - best_b)

    # Test smaller and larger values of a to minimize the difference
    while a > 0:
        b = math.ceil(n / a)
        if a * b >= n:
            diff = abs(a - b)
            if diff < min_diff:
                best_a, best_b = a, b
                min_diff = diff
        a -= 1

    return best_a, best_b


def create_mask_dictionary(masks, categories):
    # Creation of dict with masks for each category
    mask_dict = {}

    for i in range(len(categories)):
        category = categories[i]
        if category not in mask_dict:
            mask_dict[category] = []  # Initialize list if for a specific category it does not exist
        mask_dict[category].append(masks[i])  # Add masks to the specific category list

    return mask_dict


def plot_grid_masks(image, masks, categories, classes, idx):

    mask_dict = create_mask_dictionary(masks, categories)

    # Define a colormap for the masks
    cmap = ListedColormap(['none', 'blue'])  # Transparent and Red

    # Create a grid of subplots
    rows, cols = find_closest_factors(len(classes))  # Adjust based on your desired layout
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    # Loop through each class and plot
    for i in range(len(classes)):
        # Overlay the mask onto the image
        #masked_image = image.copy()
        mask = mask_dict[i]
        combined_mask = np.any(np.array(mask), axis=0)
        #masked_image[mask == 1] = [255, 0, 0]  # Example: mask overlay in red

        # Display in the corresponding grid cell
        axes[i].imshow(image)
        axes[i].imshow(combined_mask, cmap=cmap, alpha=0.5)  # Overlay mask with transparency
        axes[i].axis('off')
        axes[i].set_title(f"{classes[i]}")

    # Turn off unused subplots
    for i in range(58, len(axes)):
        axes[i].axis('off')

    # Adjust layout and show the grid
    plt.tight_layout()
    output_folder = os.path.abspath('output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, "plot_masks" + str(idx) + ".jpg"))
    plt.close()


@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    caption = retrieve_concepts(cfg.dataset.name)

    resize = 512

    train, val, test = load_classification_dataset(cfg.dataset.name, "data", resize)

    dataset = val

    torch.cuda.empty_cache()

    if cfg.modelSam == 'Florence2':
        model = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
    elif cfg.modelSam == 'GroundingDino':
        model = GroundedSam2(caption, 'Grounding DINO')

    for idx in range(100):
        print("IMG " + str(idx))
        # Get the ground truth bounding boxes and labels
        image, ground_truth_labels = dataset.__getitem__(idx)

        image = ToPILImage()(image)

        # Predict using the model
        boxes, masks, classes = model.mask_generation_with_all_concepts(image, resize)

        #print(classes)

        # print(masks.shape)

        if len(masks) > 0:
            # save_images_with_mask(boxes,masks,classes,image,model,idx) #usable in case of printing only present class mask
            # save_images_with_mask_for_all_concepts(image, masks, model, boxes)  #to print all the class masks, even if not present
            plot_grid_masks(image, masks, classes, model.ontology.classes(), idx)


if __name__ == '__main__':
    main()
