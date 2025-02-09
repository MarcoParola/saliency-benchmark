import os

import hydra
import numpy as np
import torch
import wandb

from scripts import saliency_info_generation
from src.datasets.classification import load_classification_dataset
from src.log import get_loggers
from src.utils import load_list, retrieve_concepts_ordered, retrieve_concepts_for_class
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

    if cfg.woe.dataset == True:
                # Su tutto il dataset e con tutti i concetti
                list_concepts = retrieve_concepts_ordered(cfg.dataset.name)
                list_classes = test.classes
    #
    else:
                # Only for a specified subset of classes and concepts
                list_concepts = cfg.woe.concepts
                list_classes = cfg.woe.classes

    output_folder = os.path.abspath('ablation_results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute woe score for each class

    if cfg.woe.concept_favor_against == True:
        # In this case we are analyzing for each class only a subset of concepts that are balanced in concept in favor of the class and concept against the class
        woe_score_positive = torch.zeros(len(test.classes))
        woe_score_negative = torch.zeros(len(test.classes))

        for label in list_classes:
            list_concepts_pos, list_concepts_neg = retrieve_concepts_for_class(label, dataset_name)

            score_class_pos, woe_matrix_pos = woe.compute_score(list_concepts_pos, list_classes)
            score_class_neg, woe_matrix_neg = woe.compute_score(list_concepts_neg, list_classes)

            woe_score_positive[test.classes.index(label)] = score_class_pos[test.classes.index(label)]
            woe_score_negative[test.classes.index(label)] = score_class_neg[test.classes.index(label)]

        file_name = os.path.join(output_folder,
                                     f"ablation_study_result_{cfg.woe.concept_presence_method}_favor_against_concepts_pos_neg_scores_concepts_after_intel_problem.txt")
        with open(file_name, "a") as file:
                file.write(
                    "------------------------------------------------TEST----------------------------------------------------------------------------------\n")
                file.write("Dataset:" + str(cfg.dataset.name) + "\n")
                file.write("All classes and all concept used? " + str(cfg.woe.dataset) + "\n")
                file.write("Modello:" + str(cfg.model) + "\n")
                file.write("Saliency method:" + str(cfg.saliency.method) + "\n")
                file.write("Modello SAM:" + str(cfg.modelSam) + "\n")
                file.write("Woe_score favor concepts:" + str(woe_score_positive) + "\n")
                file.write("Woe_score against concepts:" + str(woe_score_negative) + "\n")
                file.write("Woe score media favor concepts:" + str(woe_score_positive.mean().item()) + "\n")
                file.write("Woe score media against concepts:" + str(woe_score_negative.mean().item()) + "\n")
    else:
        # In this case we are considering the same set of concepts, specified before, for all the classes considered
        woe_score, woe_matrix = woe.compute_score(list_concepts, list_classes)

        output_dir = os.path.abspath('ablation_results')
        csv_woe_path = os.path.join(output_dir, model, cfg.modelSam, cfg.woe.concept_presence_method, f"woe_{cfg.saliency.method}_{dataset_name}.csv")
        if not os.path.exists(os.path.join(output_dir, model, cfg.modelSam,cfg.woe.concept_presence_method)):
            os.makedirs(os.path.join(output_dir, model, cfg.modelSam,cfg.woe.concept_presence_method))
        woe_matrix.to_csv(csv_woe_path)

        file_name = os.path.join(output_folder,f"ablation_study_result_{cfg.woe.concept_presence_method}.txt")

        with open(file_name, "a") as file:
            file.write(
                "------------------------------------------------TEST----------------------------------------------------------------------------------\n")
            file.write("Dataset:" + str(cfg.dataset.name) + "\n")
            file.write("All classes and all concept used? " + str(cfg.woe.dataset) +"\n")
            file.write("Modello:" + str(cfg.model) + "\n")
            file.write("Saliency method:" + str(cfg.saliency.method) + "\n")
            file.write("Modello SAM:" + str(cfg.modelSam) + "\n")
            file.write("Woe_score:" + str(woe_score) + "\n")
            file.write("Woe score media:" + str(woe_score.mean().item()) + "\n")

if __name__ == '__main__':
    main()