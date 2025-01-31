import numpy as np
import torch


# Thresholding in order to see how many times the computed presence score overcome a threshold, without human setting of it
def thresholding(presence, step = 0.1):
    count_overperform = 0
    count_step = 0

    for i in np.arange(0.1,1.0,step):
        if presence > i:
            count_overperform += 1
        count_step +=1

    return count_overperform / count_step

def concept_average_saliency(mask, saliency):
    if mask.shape != saliency.shape:
        print("ERROR in shape of tensors")

    size_mask = torch.sum(mask)
    print("Computation cas")

    #Avoid division by zero, if a concept is not present, return 0
    if size_mask == 0:
        return 0.0

    # Compute integral, that is equal to the sum of the scalar product between the two tensors
    weighted_sum = torch.sum(mask * saliency)
    # Normalize by size of the mask
    cas = weighted_sum / size_mask

    return thresholding(cas)

def intersection_over_union(mask, saliency):
    if mask.shape != saliency.shape:
        print("ERROR in shape of tensors")

    size_mask = torch.sum(mask)

    #Avoid division by zero, if a concept is not present, return 0
    if size_mask == 0:
        return 0.0

    # Here I have to do thresholding inside, because the threshold for binarization has to be the same as the threshold for decide presence or absence of a concept

    count_overperform = 0
    count_step = 0

    for i in np.arange(0.1, 1.0, 0.1):
        # Binarization of saliency
        binarize_saliency = saliency > i

        iou = (torch.sum(np.logical_and(binarize_saliency, mask)) / torch.sum(np.logical_or(binarize_saliency, mask)))

        if iou > i:
            count_overperform += 1
        count_step += 1

        print(f"presence: {iou}")
    print(f"superamenti: {count_overperform}")
    print(f"steps: {count_step}")

    return count_overperform / count_step

def concept_average_saliency_with_penalty(mask, saliency):
    if mask.shape != saliency.shape:
        print("ERROR in shape of tensors")

    size_mask = torch.sum(mask)
    print(size_mask)

    #Avoid division by zero, if a concept is not present, return 0
    if size_mask == 0:
        return 0.0

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

    return thresholding(casp)


def concept_presence_compute(method, mask, saliency):

    if method == 'cas':
        return concept_average_saliency(mask, saliency)
    elif method == 'iou':
        return intersection_over_union(mask, saliency)
    elif method == 'casp':
        return concept_average_saliency_with_penalty(mask, saliency)
    else:
        print("Wrong concept average method inserted")
