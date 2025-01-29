import csv
import os
from math import log

import hydra
import numpy as np
import pandas as pd
import torch

from src.datasets.classification import load_classification_dataset
from src.utils import retrieve_concepts, load_list, load_mask, load_saliency_map, retrieve_concepts_ordered


def concept_average_saliency(mask, saliency):
    if mask.shape != saliency.shape:
        print("ERROR in shape of tensors")

    size_mask = torch.sum(mask)

    #Avoid division by zero, if a concept is not present, return 0
    if size_mask == 0:
        return torch.tensor(0.0)

    # Compute integral, that is equal to the sum of the scalar product between the two tensors
    weighted_sum = torch.sum(mask * saliency)
    # Normalize by size of the mask
    cas = weighted_sum / size_mask

    return cas

def intersection_over_union(mask, saliency):
    if mask.shape != saliency.shape:
        print("ERROR in shape of tensors")

    size_mask = torch.sum(mask)

    #Avoid division by zero, if a concept is not present, return 0
    if size_mask == 0:
        return torch.tensor(0.0)

    # prendere saliency, binarizzarla, ora sono entrambe binarie (mask e saliency) e ci faccio intersection over union

    binarize_saliency = saliency > 0.5

    iou = np.logical_and(binarize_saliency, mask)/ np.logical_or(binarize_saliency, mask)

    return iou

def concept_average_saliency_with_penalty(mask, saliency):
    if mask.shape != saliency.shape:
        print("ERROR in shape of tensors")

    size_mask = torch.sum(mask)
    print(size_mask)

    #Avoid division by zero, if a concept is not present, return 0
    if size_mask == 0:
        return torch.tensor(0.0)

    mask_not = np.logical_not(mask)

    size_mask_not = torch.sum(mask_not)

    # Compute integral, that is equal to the sum of the scalar product between the two tensors
    weighted_sum = torch.sum(mask * saliency)
    # Normalize by size of the mask
    cas = weighted_sum / size_mask

    # Compute integral between saliency and mask_not
    penalization_sum = torch.sum(mask_not * saliency)

    # Normalize by size of the mask_not

    penalization_not = penalization_sum / size_mask_not

    casp = cas - penalization_not

    return casp



def compute_single_woe_score(P_h_cp, P_h):

    P_hneg_cp = 1 - P_h_cp

    P_hneg = 1 - P_h

    if P_h_cp == 0:
        woe_h_cp = 0
    else:
        if P_h_cp == 1:
            woe_h_cp = 1  # TODO is correct?
        else:
            if P_h != 0:
                woe_h_cp = log(P_h_cp / P_hneg_cp) - log(P_h / P_hneg)
            else:
                woe_h_cp = log(P_h_cp / P_hneg_cp)
    print("Woe_score_computed:", woe_h_cp)
    return woe_h_cp

class WeightOfEvidence:

    def __init__(self, dataset, dataset_name, model, saliency_method, extractor_method):
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

        #self.list_concepts = load_list(os.path.join(dir, 'list_concepts.txt'))
        self.list_concepts = retrieve_concepts_ordered(dataset_name)

        self.dir_mask = os.path.join(dir, 'masks')
        #self.dir_concept = os.path.join(dir, 'concepts')

        # Check if the tensor with the probabilities has been already computed
        output_dir = os.path.abspath('woe_prob')

        #tensor_prob_path = os.path.join(output_dir,model,extractor_method,f"woe_prob_{saliency_method}_{dataset_name}.pth")
        csv_prob_path = os.path.join(output_dir,model,extractor_method,f"woe_prob_{saliency_method}_{dataset_name}.csv")
        if os.path.exists(csv_prob_path):
            self.final_prob = pd.read_csv(csv_prob_path, index_col=0)
            print("Probability already present")
            print(self.final_prob)
        else:

            # Compute P(h|cp) for each concept cp and for each class h, and saved each element in the matrix prob
            # prob inizializzarlo in pandas dataFrame e salvare in csv e non pth
            # prob = torch.zeros(len(self.list_concepts),
            #                   len(self.dataset.classes))  #for each concept I will produce the WoE(h|cp)

            prob = pd.DataFrame(np.zeros((len(self.list_concepts), len(self.dataset.classes))),
                                     index=self.list_concepts,
                                     columns=self.dataset.classes)

            for idx in range(self.dataset.__len__()):
                print("IMG " + str(idx))
                # Get the ground truth bounding boxes and labels
                image, label = self.dataset.__getitem__(idx)
                print(f"label: {label}")

                # Retrieve mask
                masks = load_mask(os.path.join(self.dir_mask, f'mask_{idx}.pth'))
                #concepts = load_mask(os.path.join(self.dir_concept, f'classes_{idx}.pth'))
                print(f"maschera shape:{masks.shape}")
                #print(concepts.shape)
                # Retrieve saliency
                row_sal_info = self.saliency_info.iloc[idx]
                path_sal = row_sal_info[self.saliency_method+" path"]
                saliency = load_saliency_map(path_sal)

                # Compute concept presence within the image
                for i, mask in enumerate(masks):
                    concept = self.list_concepts[i]
                    #Apply thresholding to concept presence computed
                    concept_presence = 1 if concept_average_saliency(mask, saliency) > 0.5 else 0

                    prob.loc[concept, self.dataset.classes[label]] += concept_presence  #


            # Compute probability of P(h|cp)
            self.final_prob = pd.DataFrame(np.zeros((len(self.list_concepts), len(self.dataset.classes))),
                                     index=self.list_concepts,
                                     columns=self.dataset.classes)
            for label in range(prob.shape[1]):
                for concept in range(prob.shape[0]):
                    # Divide each member of the matrix for the number of times in which the concept is present in an image
                    if prob.iloc[concept].sum()!=0:
                        self.final_prob.iloc[concept, label] = prob.iloc[concept, label] / prob.iloc[concept].sum()
                    else:
                        print(prob.iloc[concept, label])
                        print(prob.iloc[concept])

            if not os.path.exists(os.path.join(output_dir,model,extractor_method)):
                os.makedirs(os.path.join(output_dir,model,extractor_method))
            #torch.save(prob, tensor_prob_path)
            self.final_prob.to_csv(csv_prob_path)

    def compute_score(self, concepts_specified, labels_specified):

        # compute total score for each class specified by the user
        woe_score = torch.zeros(len(labels_specified))

        for label_id, label in enumerate(self.dataset.classes):

            if label not in labels_specified:
                print(f"label {label} not in list {labels_specified}")
                continue

            woe_label = 0
            for concept_id, concept in enumerate(self.list_concepts):

                if concept not in concepts_specified:
                    print(f"concept {concept} not in list {concepts_specified}")
                    continue

                P_h_cp = self.final_prob.iloc[concept_id, label_id]

                # Count of number of images belonging to class "label" in the dataset and then divided
                # by total number of image, to retrieve the probability to belong to a certain class
                P_h = self.saliency_info["True Class"].value_counts().iloc[label_id] / self.dataset.__len__()
                woe_label = woe_label + compute_single_woe_score(P_h_cp, P_h)

            woe_score[label_id] = woe_label
        return woe_score





if __name__ == "__main__":
    dataset_name = "intel_image"
    model = "VGG11_Weights.IMAGENET1K_V1"
    train, val, test = load_classification_dataset(dataset_name, '', 224)

    saliency_methods = ['gradcam', 'sidu', 'lime']

    woe_gradcam = WeightOfEvidence(test, dataset_name, model, saliency_methods[0], 'GroundingDino')
    #woe_lrp = WeightOfEvidence(test, dataset_name, model, 'lrp', 'GroundingDino')
    woe_sidu = WeightOfEvidence(test, dataset_name, model, saliency_methods[1], 'GroundingDino')
    woe_lime = WeightOfEvidence(test, dataset_name, model, saliency_methods[2], 'GroundingDino')

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

    woe_stack = np.stack([woe_gradcam_score,woe_sidu_score,woe_lime_score])

    max_indices = np.argmax(woe_stack, axis=0)

    largest_values_row = woe_stack[max_indices, np.arange(woe_stack.shape[1])]

    # Output the largest rows for each column
    for idx, row in enumerate(largest_values_row):
        print(f"For classes {list_classes[idx]}: The largest value is {row} from"
              f" the saliency method: {saliency_methods[max_indices[idx]]}")








