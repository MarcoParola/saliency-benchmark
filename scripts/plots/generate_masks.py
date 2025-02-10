import json
import os

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset
from src.models.detector.grounded_sam2 import GroundedSam2
from src.utils import save_mask, load_mask, retrieve_concepts, save_list, save_images_with_mask_for_all_concepts

if GlobalHydra().is_initialized():
    GlobalHydra().clear()

# Script designed to generate the masks for a single image of the specified dataset, firstly for debug purposes

@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    caption = retrieve_concepts(cfg.dataset.name)

    resize = 224

    train, val, test = load_classification_dataset(cfg.dataset.name, "data", resize)

    dataset = test

    # Put here the number of image desired
    image_id_desired = 2996

    if cfg.modelSam == 'Florence2':
        model = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
    elif cfg.modelSam == 'GroundingDino':
        model = GroundedSam2(caption, 'Grounding DINO')

    for idx in range(image_id_desired, image_id_desired+1):
        print("IMG " + str(idx))
        # Get the ground truth bounding boxes and labels
        image, ground_truth_labels = dataset.__getitem__(idx)

        image = ToPILImage()(image)

        # Predict using the model
        boxes, masks, classes = model.mask_generation_with_all_concepts(image, resize)

        if len(masks) > 0:
            save_images_with_mask_for_all_concepts(image, masks, classes, model.ontology.classes(), boxes,0,os.path.abspath('mask_output'))  #to print all the class masks, even if not present


if __name__ == '__main__':
    main()

