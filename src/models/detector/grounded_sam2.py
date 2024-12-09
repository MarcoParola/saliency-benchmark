import json
import os
import random

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from autodistill.utils import plot
import cv2
from autodistill.detection import CaptionOntology
import supervision as sv
from autodistill_grounded_sam_2 import GroundedSAM2
from matplotlib import pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from segment_anything import sam_model_registry
from supervision.draw.color import ColorPalette
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


from src.utils import save_annotated_images

OUTPUT_DIR = "../../../images"

import requests


def get_wikipedia_description(word):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{word}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        # Extract the summary of the page
        description = data.get('extract', 'No description found.')
        return description
    else:
        return "Sorry, no description found."+chr(random.randint(32,126))

def create_ontology_from_string(caption):
    print("caption:" + str(caption))
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(caption)  #process the caption

    parts=caption.split("/")

    # Extract all nouns (both singular and plural)
    nouns = list(set([token for token in parts]))  # lemma_ is used to convert plurals in singular

    print("Nouns:")
    print('. '.join(nouns))

    #Create dictionary for ontology mapping each nouns in itself
    ontology_dict = {word: word for word in nouns}
    #ontology_dict = {get_wikipedia_description(word):word for word in nouns}
    print(ontology_dict)
    return CaptionOntology(ontology_dict)


def retrieve_labels(param, class_id):
    labels = []
    for cl in class_id:
        labels.append(param[cl])
    return labels


def save_annotated_images_grounded(model, image, results):
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=results)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=results,
                                               labels=retrieve_labels(model.ontology.classes(), results.class_id))

    # Assuming 'annotated_frame' is a PIL Image object
    if isinstance(annotated_frame, Image.Image):
        annotated_frame = np.array(annotated_frame)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=results)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)
    # label all images in a folder called `context_images`
    #model.label("../../../images", extension=".jpeg")


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class GroundedSam2(nn.Module):
    def __init__(self, prompt, model_name):
        super(GroundedSam2, self).__init__()
        print("model: " + model_name)

        self.model = GroundedSAM2(
            ontology=create_ontology_from_string(prompt),
            model=model_name,  # Choose "Florence 2" or "Grounding DINO"
            grounding_dino_box_threshold=0.4,
            grounding_dino_text_threshold=0.3
        )

        self.ontology = self.model.ontology

    def forward(self, image):
        # run inference on a single image
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = self.model.predict(image)

        #print(results)
        #print(self.model.ontology.classes())

        # if debug is True:
        #save_annotated_images_grounded(self.model, image, results)

        bbox = results.xyxy
        categories = results.class_id
        confidence_score = results.confidence


        keep_indices = torch.ops.torchvision.nms(torch.tensor(bbox), torch.tensor(confidence_score), iou_threshold=0.5)
        # print(keep_indices)

        bbox = bbox[keep_indices]
        categories = categories[keep_indices]
        confidence_score = confidence_score[keep_indices]


        if bbox.ndim == 1:  #se dopo NMS resta una sola box
            bbox = np.expand_dims(bbox, axis=0)
            categories = np.array([categories])
            confidence_score = np.array([confidence_score])

        # results = sv.Detections(
        #     xyxy=bbox,
        #     confidence=confidence_score,
        #     class_id=categories
        # )
        #
        # labels = [self.model.ontology.classes()[cat] for cat in categories]

        # annotated_frame=save_annotated_images(labels, image, results)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

        return bbox, categories, confidence_score


    def automatic_mask_generation(self, image):

        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        #USANDO REPO SAM2

        sam2_checkpoint = "~/.cache/autodistill/segment_anything_2/sam2_hiera_large.pt"
        checkpoint = os.path.expanduser(sam2_checkpoint)
        model_cfg = "sam2_hiera_l.yaml"
        
        sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
        print(self.model.sam_2_predictor.model)
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2)

        #USANDO REPO SAM_ANYTHING

        # sam_checkpoint = "sam_vit_h_4b8939.pth"
        # model_type = "vit_h"
        #
        # device = "cuda"
        #
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)
        #
        # mask_generator = SamAutomaticMaskGenerator(sam)
        #mask_generator = SamAutomaticMaskGenerator(self.model.sam_2_predictor.model)
        #print(mask_generator)

        masks = mask_generator.generate(image)
        print(masks)
        print(len(masks))
        print(masks[0].keys())
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    caption = "cat/chicken/cow/dog/fox/goat/horse/person/racoon/skunk"

    IMAGE_PATH = "images/buildings.jpg"

    torch.cuda.empty_cache()

    model = GroundedSam2(caption, "Grounding DINO")

    #torch.cuda.empty_cache()

    #bbox, categories, confidence = model(cv2.imread(IMAGE_PATH))

    image = Image.open(IMAGE_PATH)
    image = np.array(image.convert("RGB"))

    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    model.automatic_mask_generation(image)
    # print(bbox)
    # print(categories)
    # print(confidence)
