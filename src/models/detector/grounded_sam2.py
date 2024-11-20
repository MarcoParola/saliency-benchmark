import json
import os

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.utils import plot
import cv2
import spacy
from autodistill.detection import CaptionOntology
import supervision as sv
from supervision.draw.color import ColorPalette

from src.utils import save_annotated_images

OUTPUT_DIR = "../../../output"

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
        return "Sorry, no description found."

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
        # print(bbox)
        # print(categories)
        # print(confidence_score)

        keep_indices = torch.ops.torchvision.nms(torch.tensor(bbox), torch.tensor(confidence_score), iou_threshold=0.5)
        # print(keep_indices)

        bbox = bbox[keep_indices]
        categories = categories[keep_indices]
        confidence_score = confidence_score[keep_indices]
        # print(bbox)
        # print(categories)
        # print(confidence_score)

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


if __name__ == '__main__':
    caption = "cat chicken cow dog fox goat horse person racoon skunk"

    IMAGE_PATH = "../../../images/horse.jpg"

    torch.cuda.empty_cache()

    model = GroundedSam2(caption)

    torch.cuda.empty_cache()

    bbox, categories, confidence = model(cv2.imread(IMAGE_PATH))
    # print(bbox)
    # print(categories)
    # print(confidence)
