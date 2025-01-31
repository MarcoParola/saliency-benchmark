import os

import hydra
import numpy as np
import wandb

from scripts import saliency_info_generation
from src.datasets.classification import load_classification_dataset
from src.log import get_loggers
from src.utils import load_list, retrieve_concepts_ordered
from src.woe import WeightOfEvidence


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):

    dataset_name = cfg.dataset.name
    model = cfg.model
    # Load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(dataset_name, data_dir, cfg.dataset.resize)

    # Retrieve saliency maps info

    output_dir_csv = os.path.join(os.path.abspath('saliency_info'), model)

    output_csv = os.path.join(output_dir_csv, f'saliency_info_{dataset_name}.csv')

    if not os.path.exists(output_csv):
        saliency_info_generation.main(cfg) # It will be read in the WeightOfEvidence class

    woe = WeightOfEvidence(test, dataset_name, model, cfg.saliency.method, cfg.modelSam, cfg.woe.concept_presence_method)

    list_concepts = retrieve_concepts_ordered(cfg.dataset.name)

    list_classes = test.classes

    # Compute woe score for each class

    woe_score = woe.compute_score(list_concepts, list_classes)

    output_folder = os.path.abspath('ablation_results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.join(output_folder,f"ablation_study_result_{cfg.woe.concept_presence_method}.txt")

    with open(file_name, "a") as file:
        print("printing on file")
        file.write(
            "------------------------------------------------TEST----------------------------------------------------------------------------------\n")
        file.write("Dataset:" + str(cfg.dataset.name) + "\n")
        file.write("Modello:" + str(cfg.model) + "\n")
        file.write("Saliency method:" + str(cfg.saliency.method) + "\n")
        file.write("Modello SAM:" + str(cfg.modelSam) + "\n")
        file.write("Woe_score:" + str(woe_score) + "\n")
        file.write("Woe score media:" + str(woe_score.mean().item()) + "\n")

if __name__ == '__main__':
    main()