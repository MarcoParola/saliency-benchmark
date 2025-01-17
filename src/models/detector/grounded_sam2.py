import json
import math
import os
import random

import numpy
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
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset, ClassificationDataset
from src.datasets.detection import load_detection_dataset
from src.utils import save_annotated_images, save_mask
from matplotlib.colors import ListedColormap

OUTPUT_DIR = "output"

import requests

def create_ontology_from_string(caption):
    print("caption:" + str(caption))
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(caption)  #process the caption

    parts = caption.split("/")

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


def save_images_with_mask(input_boxes, masks, class_ids, img, model, idx):
    img = np.array(img)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    #box_annotator = sv.BoxAnnotator()
    #annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=img, detections=detections,
                                               labels=retrieve_labels(model.ontology.classes(), detections.class_id))

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    image = annotated_frame.astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask" + str(idx) + ".jpg"), image_rgb)

def generate_mask_for_all_concepts(model,classes, masks, boxes, resize):

    if len(masks)>0:
        for i in range(len(model.ontology.classes())):
            if i not in classes:
                masks = np.append(masks, np.expand_dims(np.full((resize, resize), False, dtype=bool), axis=0), axis=0)
                boxes = np.append(boxes, np.expand_dims([0, 0, 0, 0], axis=0), axis=0)
                classes = np.append(classes, i)
    else:
        masks = np.expand_dims(np.full((resize, resize), False, dtype=bool), axis=0)
        masks = np.repeat(masks, len(model.ontology.classes()), axis=0)
        boxes = np.expand_dims([0, 0, 0, 0], axis=0)
        boxes = np.repeat(boxes, len(model.ontology.classes()), axis=0)
        classes = np.arange(len(model.ontology.classes()))

    masks = np.stack(masks)
    boxes = np.stack(boxes)
    classes = np.stack(classes)

    return masks, boxes, classes

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

    def mask_generation(self, image):  #function which generate a mask only for the concept present in the image
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = self.model.predict(image)

        boxes = results.xyxy

        masks = results.mask

        classes = results.class_id

        confidence_score = results.confidence

        keep_indices = torch.ops.torchvision.nms(torch.tensor(boxes), torch.tensor(confidence_score), iou_threshold=0.5)
        # print(keep_indices)

        boxes = boxes[keep_indices]
        classes = classes[keep_indices]
        masks = masks[keep_indices]

        if boxes.ndim == 1:  # se dopo NMS resta una sola box
            boxes = np.expand_dims(boxes, axis=0)
            classes = np.array([classes])
            masks = np.expand_dims(masks, axis=0)

        return boxes, masks, classes

    def mask_generation_with_all_concepts(self, image, resize): #function which generate a mask (with all values to false) also for the concept not present in the image

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = self.model.predict(image)

        boxes = results.xyxy

        masks = results.mask

        classes = results.class_id

        confidence_score = results.confidence

        keep_indices = torch.ops.torchvision.nms(torch.tensor(boxes), torch.tensor(confidence_score), iou_threshold=0.8)
        # print(keep_indices)

        boxes = boxes[keep_indices]
        classes = classes[keep_indices]
        masks = masks[keep_indices]

        if boxes.ndim == 1:  # se dopo NMS resta una sola box
            boxes = np.expand_dims(boxes, axis=0)
            classes = np.array([classes])
            masks = np.expand_dims(masks, axis=0)

        masks, boxes, classes = generate_mask_for_all_concepts(self.model,classes, masks, boxes,resize)

        return boxes, masks, classes


if __name__ == '__main__':
    caption = ("Fin/Ears/Muzzle/Paws/Tail/Body/Button/Speaker/Handle/Blade"
                "/Rose window/Facade/Bell/Cab/Wheel/Hose/Nozzle/Tank/Golf ball dimples/Logo/Canopy")  #26 concepts for Imagenette
    #caption = "Wall/Window/Roof/Façade/Tree/Vegetation/Rock/Ice/Mountain Peak/Beach/Boat/Water/Sidewalk/Streetlights/Car/Sky" #16 concepts for Intel_Image

    # caption = ("Very defined erythematous edges/Uneven grey ulcer ground/Granulating and necrotic ulcer background/ "
    #            "White Fibrin/ Marked erythematous edges")
    #IMAGE_PATH = "images/flower.jpg"

    resize = 512

    train, val, test = load_classification_dataset("imagenette", "data", resize)

    dataset = val

    torch.cuda.empty_cache()

    model = GroundedSam2(caption, "Grounding DINO")

    print(model.ontology.classes())

    # Mostra il path assoluto del file
    absolute_path = os.path.abspath(OUTPUT_DIR)
    print(f"Il file è stato salvato in: {absolute_path}")

    image, label = dataset.__getitem__(1000)

    image = ToPILImage()(image)

    #bbox, categories, confidence = model(image)

    #print(bbox)
    #print(categories)
    #print(confidence)

    bbox, masks, categories = model.mask_generation_with_all_concepts(image, resize)

    #print(bbox)
    print(categories)
    #print(masks)

    #save_images_with_mask_for_all_concepts(image, masks, categories, bbox, 331)

    #plot_grid_masks(image, masks, categories, model.ontology.classes(), 1000)

    #bbox, masks, categories = model.mask_generation(image)

    #print(bbox)
    #print(categories)
    #print(masks)

    save_images_with_mask(bbox,masks,categories,image,model, 1000)


    
