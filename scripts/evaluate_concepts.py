import os

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.datasets.classification import load_classification_dataset
from src.utils import load_mask, load_list


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    print("evaluating concepts...")
    absolute_path = os.path.abspath("mask_output")

    dir = os.path.join(absolute_path, cfg.dataset.name)

    list_concepts = load_list(os.path.join(dir, 'list_classes.txt'))
    print(list_concepts)
    n_concepts = len(list_concepts)

    n_classes = cfg[cfg.dataset.name].n_classes
    print(n_classes)
    print(n_concepts)
    mat = torch.zeros(n_concepts, n_classes)
    print(mat)
    print(mat.shape)

    train, val, test = load_classification_dataset(cfg.dataset.name, 'data', 224)

    dataset = test
    print("Classes dataset:", dataset.classes)

    counter_classes = torch.zeros(n_classes)

    # Construct the matrix with each percentage
    for i in range(dataset.__len__()):
        masks = load_mask(os.path.join(dir,'masks',f'mask_{i}.pth'))
        concepts = load_mask(os.path.join(dir,'classes',f'classes_{i}.pth'))
        image,label = dataset.__getitem__(i)
        counter_classes[label] += 1
        for mask,concept in zip(masks,concepts):
            concept = int(concept.item())
            cont_pixels = torch.sum(mask)/(cfg.dataset.resize*cfg.dataset.resize)
            mat[concept,label] += cont_pixels

    # Divide each element of the matrix for the number of images belonging to that specific class
    for j in range(mat.shape[1]):
        if counter_classes[j] != 0:
            mat[:, j] = mat[:, j] / counter_classes[j]

    # Generate histogram for each class
    num_rows = len(dataset.classes)//2
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 15), sharey=True)

    axes = axes.flatten()  # to make 1D the axes array

    for col_idx, col_label in enumerate(dataset.classes):
        values = mat[:, col_idx]
        axes[col_idx].bar(list_concepts, values, color='skyblue', edgecolor='black')
        axes[col_idx].set_title(col_label)
        #axes[col_idx].set_xlabel('Rows')
        if col_idx % 2 == 0:  # values of axes y only for the right subplot
            axes[col_idx].set_ylabel('Values')
        axes[col_idx].tick_params(axis='x', labelrotation=45)

    for ax in axes[len(dataset.classes):]:  # to make inactive the empty axes
        ax.axis('off')

    plt.tight_layout()
    #plt.show()
    output_dir = os.path.abspath('histograms')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "hist_" + cfg.dataset.name + ".jpg"))
    plt.close()

if __name__ == "__main__":
    main()
