import os

import hydra
import numpy as np

from src.datasets.classification import load_classification_dataset
from src.utils import load_list
from src.woe import WeightOfEvidence


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):

    dataset_name = cfg.dataset.name
    model = cfg.model
    # Load test dataset
    data_dir = os.path.join(cfg.mainDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(dataset_name, data_dir, cfg.dataset.resize)

    saliency_methods = ['gradcam', 'sidu', 'lime']

    woe_gradcam = WeightOfEvidence(test, dataset_name, model, saliency_methods[0], 'GroundingDino')
    # woe_lrp = WeightOfEvidence(test, dataset_name, model, 'lrp', 'GroundingDino')
    woe_sidu = WeightOfEvidence(test, dataset_name, model, saliency_methods[1], 'GroundingDino')
    woe_lime = WeightOfEvidence(test, dataset_name, model, saliency_methods[2], 'GroundingDino')

    if cfg.woe.dataset == True:

        # Su tutto il dataset
        absolute_path = os.path.abspath("mask_output")

        dir = os.path.join(absolute_path, dataset_name)

        list_concepts = load_list(os.path.join(dir, 'list_concepts.txt'))

        list_classes = test.classes

    else:
        # Only for a specified subset of classes and concepts
        list_concepts = cfg.woe.concepts
        list_classes = cfg.woe.classes

    # Compute woe score for each class
    print(f"list concepts: {list_concepts}")
    print(f"list classes: {list_classes}")

    woe_gradcam_score = woe_gradcam.compute_score(list_concepts, list_classes)
    # woe_lrp_score = compute_total_woe_score(woe_lrp, list_concepts, list_classes)
    woe_sidu_score = woe_sidu.compute_score(list_concepts, list_classes)
    woe_lime_score = woe_lime.compute_score(list_concepts, list_classes)

    print(woe_gradcam_score)
    print(woe_sidu_score)
    print(woe_lime_score)

    # Now for each class we can see which is the best saliency method for the specified set of concepts

    woe_stack = np.stack([woe_gradcam_score, woe_sidu_score, woe_lime_score])

    max_indices = np.argmax(woe_stack, axis=0)

    largest_values_row = woe_stack[max_indices, np.arange(woe_stack.shape[1])]

    # Output the largest rows for each column
    for idx, row in enumerate(largest_values_row):
        print(f"For classes {list_classes[idx]}: The largest value is {row} from"
              f" the saliency method: {saliency_methods[max_indices[idx]]}")


if __name__ == '__main__':
    main()
