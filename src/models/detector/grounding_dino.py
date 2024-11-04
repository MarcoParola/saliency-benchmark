import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from importlib import resources
import torch.nn as nn
import torch
import spacy
import os
import urllib.request
import supervision as sv
from supervision import ColorPalette
from torchvision.ops import box_convert
from typing import Tuple, List

OUTPUT_DIR = "..\..\..\..\output"

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
# IMAGE_PATH = "weights/dog-3.jpeg"
# TEXT_PROMPT = "chair . person . dog ."
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25
#
# image_source, image = load_image(IMAGE_PATH)
#
# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )
#
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)

def create_prompt_from_caption(text):
    print("caption:" + str(text))
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(caption)  # process the caption

    # Extract all nouns (both singular and plural)
    nouns = list(set([token.lemma_ for token in doc if
                      token.pos_ in ["NOUN", "PROPN"]]))  # lemma_ is used to convert plurals in singular

    print("Nouns:")
    prompt = '. '.join(nouns)
    print(prompt)

    return prompt


def create_prompt_from_classes(text):
    parts = text.split("/")

    # Extract all nouns (both singular and plural)
    nouns = list(set([token for token in parts]))  # lemma_ is used to convert plurals in singular

    print("Nouns:")
    print('. '.join(nouns))

    prompt = '. '.join(nouns)
    return prompt


def save_annotate_grounding_dino(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor,
                                 phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy, class_id=np.arange(len(boxes)))

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    label_annotator = sv.LabelAnnotator(color=ColorPalette(colors=[sv.Color(255, 0, 0)]),
                                        text_position=sv.Position.BOTTOM_LEFT)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


class GroundingDino(nn.Module):
    def __init__(self):
        super(GroundingDino, self).__init__()

        # Define the directory and weights file path
        weights_dir = "weights"
        weights_file = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
        weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

        # Create the weights directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)

        # Download the weights file if it doesn't exist
        if not os.path.exists(weights_file):
            print("Downloading GroundingDINO weights...")
            urllib.request.urlretrieve(weights_url, weights_file)
            print("Download complete.")
        else:
            print("Weights file already exists. Skipping download.")

        # Access model configuration and weights directly from the package
        with resources.path("groundingdino.config", "GroundingDINO_SwinT_OGC.py") as config_path, \
                resources.path("weights", "groundingdino_swint_ogc.pth") as weights_path:
            self.model = load_model(str(config_path), str(weights_path))

    def forward(self, image_path, prompt):
        image_source, image = load_image(image_path)
        print(image.type())

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=create_prompt_from_classes(prompt),
            box_threshold=0.35,
            text_threshold=0.25
        )

        annotated_frame = save_annotate_grounding_dino(image_source=image_source, boxes=boxes, logits=logits,
                                                       phrases=phrases)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "image_box" + str(iteration) + ".jpg"), annotated_frame)
        return boxes, phrases,logits


if __name__ == '__main__':
    caption = "cat chicken cow dog fox goat horse person racoon skunk"

    IMAGE_PATH = "../../../images/racoons.jpg"

    torch.cuda.empty_cache()

    model = GroundingDino()

    bbox, categories, confidence = model(IMAGE_PATH, caption)
    print(bbox)
    print(categories)
    print(confidence)
