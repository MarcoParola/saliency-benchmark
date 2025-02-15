import os

import cv2
import numpy as np
import torch
import hydra
import torchvision
from hydra.core.global_hydra import GlobalHydra
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset
from src.models.detector.grounded_sam2 import GroundedSam2
from src.utils import save_mask, load_mask, retrieve_concepts, save_list, save_images_with_mask_for_all_concepts, \
    retrieve_concepts_ordered

if GlobalHydra().is_initialized():
    GlobalHydra().clear()

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):

    if cfg.mask.dataset:
        caption = retrieve_concepts(cfg.dataset.name)
        #print(caption)
        list_ordered_concept = retrieve_concepts_ordered(cfg.dataset.name)
        #print(f"list ordered:{list_ordered_concept}")

        resize = 224

        train, val, test = load_classification_dataset(cfg.dataset.name, "data", resize)

        dataset = test

        torch.cuda.empty_cache()

        if cfg.modelSam == 'Florence2':
            model = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
        elif cfg.modelSam == 'GroundingDino':
            model = GroundedSam2(caption, 'Grounding DINO')

        output_folder = os.path.join(os.path.abspath('mask_output'), cfg.modelSam, cfg.dataset.name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for idx in range(dataset.__len__()):
            print("IMG " + str(idx))
            # Get the ground truth bounding boxes and labels
            image, ground_truth_labels = dataset.__getitem__(idx)

            image = ToPILImage()(image)

            # Predict using the model
            boxes, masks, classes = model.mask_generation_with_all_concepts(image, resize)

            # print(masks.shape)

            if len(masks) > 0:

                output_dir_tensors = os.path.join(output_folder,'masks')
                if not os.path.exists(output_dir_tensors):
                    os.makedirs(output_dir_tensors)

                #mi carico la lista ordinata di concetti, for su questo, vedo se è presente su classes, allora lo associo, riordina
                #ndo la maschera
                print(f"maschere: {masks}")
                print(f"concepts: {classes}")

                masks_ordered = np.expand_dims(np.full((resize, resize), False, dtype=bool), axis=0)
                masks_ordered = np.repeat(masks_ordered, len(list_ordered_concept), axis=0)
                class_ordered = np.zeros(len(list_ordered_concept))

                list_concept_ontology = model.ontology.classes()

                for i, concept_ordered in enumerate(list_ordered_concept):
                    print(f"concept: {concept_ordered} in index {i}")
                    # make bind between ordered concept and concept of GroundedSam2, finding the class corresponding to the concept of this cycle for
                    idx_concept = list_concept_ontology.index(concept_ordered)

                    for mask, concept in zip(masks, classes):
                        id = int(concept.item())
                        if id == idx_concept:
                            masks_ordered[i] = np.logical_or(masks_ordered[i], mask)

                    # masks_ordered = np.append(masks_ordered, mask_single_concept)
                    # masks_ordered[i] = mask_single_concept
                    # # only for debug purpose TODO delete it
                    class_ordered[i] = idx_concept
                    # class_ordered = np.append(class_ordered, concepts[idx_concept])

                print(f"maschere: {masks_ordered}")
                print(f"concepts: {class_ordered}")

                save_mask(os.path.join(output_dir_tensors, f'mask_{idx}.pth'), masks_ordered)
    else:
        # if cfg.mask.dataset == False, that means that we want to produce the mask for a single image

        caption = cfg.mask.concepts

        if cfg.modelSam == 'Florence2':
            model = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
        elif cfg.modelSam == 'GroundingDino':
            model = GroundedSam2(caption, 'Grounding DINO')

        image = cv2.imread(f"data/image/{cfg.mask.file_image}")

        resize = cfg.dataset.resize

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((cfg.dataset.resize, cfg.dataset.resize)),
        ])

        img = transform(image)
        # clip image to be three channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        image = img

        image = ToPILImage()(image)

        # Predict using the model
        boxes, masks, classes = model.mask_generation_with_all_concepts(image, resize)

        if len(masks) > 0:
            output_dir = os.path.join(os.path.abspath("mask_output"), 'single_image')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_images_with_mask_for_all_concepts(image, masks, classes, model.ontology.classes(), boxes,0, output_dir)  #to print all the class masks, even if not present


if __name__ == '__main__':
    main()
