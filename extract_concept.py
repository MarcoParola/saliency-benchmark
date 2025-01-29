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
        print(caption)
        list_ordered_concept = retrieve_concepts_ordered(cfg.dataset.name)
        print(f"list ordered:{list_ordered_concept}")
        # IMAGE_PATH = "images/flower.jpg"

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

        save_list(os.path.join(output_folder,'list_concepts.txt'), model.ontology.classes())

        for idx in range(dataset.__len__()):
            print("IMG " + str(idx))
            # Get the ground truth bounding boxes and labels
            image, ground_truth_labels = dataset.__getitem__(idx)

            image = ToPILImage()(image)

            # Predict using the model
            boxes, masks, classes = model.mask_generation_with_all_concepts(image, resize)

            # print(masks.shape)

            if len(masks) > 0:
                # save_images_with_mask(boxes,masks,classes,image,model,idx) #usable in case of printing only present class mask
                # save_images_with_mask_for_all_concepts(image, masks, model, boxes)  #to print all the class masks, even if not present
                # plot_grid_masks(image, masks, model.ontology.classes(), idx)
                output_dir_tensors = os.path.join(output_folder,'masks')
                output_dir_classes = os.path.join(output_folder,'concepts')
                if not os.path.exists(output_dir_tensors):
                    os.makedirs(output_dir_tensors)
                if not os.path.exists(output_dir_classes):
                    os.makedirs(output_dir_classes)

                #mi carico la lista ordinata di concetti, for su questo, vedo se è presente su classes, allora lo associo, riordina
                #ndo la maschera

                masks_ordered = np.zeros(len(list_ordered_concept))
                class_ordered = np.zeros(len(list_ordered_concept))

                list_concept_ontology = model.ontology.classes()

                print("lista ontology:{list_concept_ontology}")

                for concept in list_ordered_concept:
                    print(f"concept: {concept}")
                    #make bind between ordered concept and concept of GroundedSam2, finding the class corresponding to the concept of this cycle for
                    idx_concept = list_concept_ontology.index(concept)
                    masks_ordered = np.append(masks_ordered, masks[idx_concept])
                    #only for debug purpose TODO delete it
                    class_ordered = np.append(class_ordered, classes[idx_concept])


                save_mask(os.path.join(output_dir_tensors, f'mask_{idx}.pth'), masks_ordered)
                #save_mask(os.path.join(output_dir_classes, f'classes_{idx}.pth'), classes) # I keep also the numpy array containing the classes
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
            # save_images_with_mask(boxes,masks,classes,image,model,idx) #usable in case of printing only present class mask
            save_images_with_mask_for_all_concepts(image, masks, classes, model.ontology.classes(), boxes,0, output_dir)  #to print all the class masks, even if not present
            # plot_grid_masks(image, masks, model.ontology.classes(), idx)


if __name__ == '__main__':
    main()
