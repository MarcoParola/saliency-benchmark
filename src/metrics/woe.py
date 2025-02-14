import os
from math import log

import numpy as np
import pandas as pd
import torch

from src.metrics.concept_presence import concept_presence_compute
from src.datasets.classification import load_classification_dataset
from src.utils import retrieve_concepts, load_list, load_mask, load_saliency_map, retrieve_concepts_ordered


def compute_single_woe_score(P_h_cp, P_h):
    P_hneg_cp = 1 - P_h_cp

    P_hneg = 1 - P_h

    if P_h_cp == 0:
        print("Error, Prob to 0")
    else:
        if P_h != 0:
            woe_h_cp = log(P_h_cp / P_hneg_cp) - log(P_h / P_hneg)
        else:
            woe_h_cp = log(P_h_cp / P_hneg_cp)
    return woe_h_cp


def compute_probability(occ, row_occ_concept_id, label_id, concept_id):

    #print("Row occ: ", row_occ_concept_id) # It contains the row of the specified concept, but with only the cells regarding the labels_specified

    # Compute probability of class h, in set of class labels_specified, given concept positioned in row concept_id of occurrences matrix
    occ_cp_h = occ.iloc[concept_id, label_id]

    sum_cell_cp_labels_specified = row_occ_concept_id.sum()

    P_h_cp = occ_cp_h/sum_cell_cp_labels_specified

    return P_h_cp


class WeightOfEvidence:

    def __init__(self, dataset, dataset_name, model, saliency_method, extractor_method, concept_presence_method):
        self.saliency_method = saliency_method
        self.extractor_method = extractor_method
        self.dataset = dataset  # passed as torch dataset

        # Retrieve saliency maps info

        output_dir_csv = os.path.join(os.path.abspath('saliency_info'), model)

        output_csv = os.path.join(output_dir_csv, f'saliency_info_{dataset_name}.csv')

        self.saliency_info = pd.read_csv(output_csv)

        # Retrieve caption to pass to GroundedSam2

        self.caption = retrieve_concepts(dataset_name)

        # Retrieve all extracted masks info

        absolute_path = os.path.abspath("mask_output")

        dir = os.path.join(absolute_path, extractor_method, dataset_name)

        self.list_concepts = retrieve_concepts_ordered(dataset_name)

        self.dir_mask = os.path.join(dir, 'masks')
        epsilon = 5 * (10 ** (-4)) # small epsilon added to each WoE score to avoid logarithm going to +/- infinite

        # Check if the tensor with the occurrences has been already computed
        output_dir = os.path.abspath(f'occurrence_{concept_presence_method}')

        csv_occ_path = os.path.join(output_dir, model, extractor_method, f"occ_{saliency_method}_{dataset_name}.csv")
        if os.path.exists(csv_occ_path):
            self.occ = pd.read_csv(csv_occ_path, index_col=0)
            print("Occurrences already present")
            self.occ += epsilon
        else:

            # Compute Occurrences for each concept cp and for each class h, and saved each element in the matrix self.occ

            self.occ = pd.DataFrame(np.zeros((len(self.list_concepts), len(self.dataset.classes))),
                                index=self.list_concepts,
                                columns=self.dataset.classes)

            for idx in range(self.dataset.__len__()):
                print("IMG " + str(idx))
                # Get the ground truth bounding boxes and labels
                image, label = self.dataset.__getitem__(idx)

                # Retrieve mask
                masks = load_mask(os.path.join(self.dir_mask, f'mask_{idx}.pth'))

                # Retrieve saliency
                row_sal_info = self.saliency_info.iloc[idx]
                path_sal = row_sal_info[self.saliency_method + " path"]
                saliency = load_saliency_map(path_sal)

                # Compute concept presence within the image
                for i, mask in enumerate(masks):
                    concept = self.list_concepts[i]
                    #Apply thresholding to concept presence computed
                    concept_presence = concept_presence_compute(concept_presence_method, mask, saliency)

                    self.occ.loc[concept, self.dataset.classes[label]] += concept_presence  # increment occurrences for cell (concept,class)

            if not os.path.exists(os.path.join(output_dir, model, extractor_method)):
                os.makedirs(os.path.join(output_dir, model, extractor_method))
            self.occ.to_csv(csv_occ_path)
            self.occ += epsilon

    def compute_score(self, concepts_specified, labels_specified):

        # compute total score for each class specified by the user
        woe_score = torch.zeros(len(self.dataset.classes))

        # mantain also the single score computed for each couple (concept,class)
        woe_matrix = pd.DataFrame(np.zeros((len(self.list_concepts), len(self.dataset.classes))),
                                index=self.list_concepts,
                                columns=self.dataset.classes)

        for label_id, label in enumerate(self.dataset.classes):

            if label not in labels_specified:
                #print(f"label {label} not in list {labels_specified}")
                continue

            woe_label = 0
            for concept_id, concept in enumerate(self.list_concepts):

                if concept not in concepts_specified:
                    #print(f"concept {concept} not in list {concepts_specified}")
                    continue

                P_h_cp = compute_probability(self.occ, self.occ.loc[concept, labels_specified], label_id, concept_id)

                # Count of number of images belonging to class "label" in the dataset and then divided
                # by total number of image, to retrieve the probability to belong to a certain class
                P_h = self.saliency_info["Predicted Class"].value_counts().iloc[label_id] / self.dataset.__len__()
                woe_score_concept_class = compute_single_woe_score(P_h_cp, P_h)
                woe_label = woe_label + woe_score_concept_class
                woe_matrix.loc[concept,label] = woe_score_concept_class

            woe_score[label_id] = woe_label
        return woe_score, woe_matrix


if __name__ == "__main__":
    dataset_name = "intel_image"
    model = "VGG11_Weights.IMAGENET1K_V1"
    train, val, test = load_classification_dataset(dataset_name, '', 224)

    saliency_methods = ['gradcam', 'sidu', 'lime']

    woe_gradcam = WeightOfEvidence(test, dataset_name, model, saliency_methods[0], 'GroundingDino', 'cas')
    #woe_lrp = WeightOfEvidence(test, dataset_name, model, 'lrp', 'GroundingDino')
    woe_sidu = WeightOfEvidence(test, dataset_name, model, saliency_methods[1], 'GroundingDino', 'cas')
    woe_lime = WeightOfEvidence(test, dataset_name, model, saliency_methods[2], 'GroundingDino', 'cas')

    # Su tutto il dataset
    absolute_path = os.path.abspath("mask_output")

    dir = os.path.join(absolute_path, dataset_name)

    list_concepts = load_list(os.path.join(dir, 'list_concepts.txt'))
    print(list_concepts)

    list_classes = test.classes
    print(list_classes)

    # Compute woe score for each class

    woe_gradcam_score = woe_gradcam.compute_score(list_concepts, list_classes)
    #woe_lrp_score = compute_total_woe_score(woe_lrp, list_concepts, list_classes)
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
