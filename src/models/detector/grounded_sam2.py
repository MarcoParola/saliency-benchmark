import json
import os

import torch.nn as nn
import torch
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.utils import plot
import cv2
import spacy
from autodistill.detection import CaptionOntology
import supervision as sv
from supervision.draw.color import ColorPalette

OUTPUT_DIR = "../../../output"


def create_ontology_from_string(caption):
    print("caption:" + str(caption))
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(caption)  #process the caption

    # Extract all nouns (both singular and plural)
    nouns = list(set([token.lemma_ for token in doc if
                      token.pos_ in ["NOUN", "PROPN"]]))  # lemma_ is used to convert plurals in singular

    print("Nouns:")
    print('. '.join(nouns))

    #Create dictionary for ontology mapping each nouns in itself
    ontology_dict = {word: word for word in nouns}
    return CaptionOntology(ontology_dict)


def retrieve_labels(param, class_id):
    labels = []
    for cl in class_id:
        labels.append(param[cl])
    return labels


def save_annotated_images(model, image, results):
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=results)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=results,
                                               labels=retrieve_labels(model.ontology.classes(), results.class_id))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=results)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)
    # label all images in a folder called `context_images`
    model.label("../../../images", extension=".jpeg")


class GroundedSam2(nn.Module):
    def __init__(self, prompt):
        super(GroundedSam2, self).__init__()

        self.model = GroundedSAM2(
            ontology=create_ontology_from_string(prompt),
            model="Grounding DINO",  # Choose "Florence 2" or "Grounding DINO"
            grounding_dino_box_threshold=0.4,
            grounding_dino_text_threshold=0.3
        )

    def forward(self, image):
        # run inference on a single image
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = self.model.predict(image)

        #print(results)
        # with open("../../../output/output.txt", "w") as file:
        #     file.write(str(results))
        #
        # plot(
        #     image=image,
        #     classes=self.model.ontology.classes(),
        #     detections=results
        # )

        #save_annotated_images(self.model, image, results)
        bbox=results.xyxy
        categories=results.class_id
        return bbox,categories


if __name__ == '__main__':
    caption = ("The image depicts a group of people gathered around a pool, with a green wall and a few fish swimming "
               "in the water")
    caption = ("The image depicts a woman walking along the edge of a large body of water, with a cityscape in the "
               "background. In the foreground, the woman is dressed in a brown jacket, blue jeans, and a black purse "
               "slung over her shoulder. She has her dark hair pulled back into a bun and is carrying a blue plastic "
               "bag in her right hand. A concrete barrier separates her from the water, which appears to be a lake or "
               "river, with a fountain spraying water into the air at its center. In the background, a grassy area is "
               "visible, with a row of trees lining the edge of the water. Beyond the trees, several tall buildings "
               "rise into the sky, including one with a crane on its roof. The sky above is overcast, with a few "
               "clouds visible. The overall atmosphere of the image suggests a peaceful and serene setting, "
               "with the woman enjoying a leisurely walk along the water's edge.")
    IMAGE_PATH = "../../../images/buildings.jpg"

    torch.cuda.empty_cache()

    model = GroundedSam2(caption)

    torch.cuda.empty_cache()

    bbox,categories = model(cv2.imread(IMAGE_PATH))
    print(bbox)
    print(categories)
