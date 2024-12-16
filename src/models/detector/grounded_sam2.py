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
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset, ClassificationDataset
from src.datasets.detection import load_detection_dataset
from src.utils import save_annotated_images
from matplotlib.colors import ListedColormap


OUTPUT_DIR = "output"

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
        return "Sorry, no description found." + chr(random.randint(32, 126))


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


def save_automatic_generated_mask(masks, image):
    print(masks)
    print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask" + str(idx) + ".jpg"))
    plt.close()


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


def generate_mask_for_all_concepts(classes, masks, boxes):
    dict_mask = dict(zip(classes, masks))
    dict_boxes = dict(zip(classes, boxes))

    for _, mask in dict_mask.items():
        print(mask.shape)

    for i in range(len(model.ontology.classes())):
        if i not in classes:
            dict_mask.update({i: np.full((resize, resize), False, dtype=bool)})
            dict_boxes.update({i: [0, 0, 0, 0]})

    sorted_mask = dict(sorted(dict_mask.items(), key=lambda x: x[0]))
    sorted_boxes = dict(sorted(dict_boxes.items(), key=lambda x: x[0]))
    masks_list = [mask for _, mask in sorted_mask.items()]
    boxes_list = [boxes for _, boxes in sorted_boxes.items()]

    # for _, mask in sorted_mask.items():
    #     print(mask.shape)

    masks = np.stack(masks_list)
    boxes = np.stack(boxes_list)

    print(masks.shape)
    print(boxes.shape)

    return masks, boxes


def save_images_with_mask_for_all_concepts(image, masks, model, boxes):
    img = np.array(image)
    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool),  # (n, h, w)
        class_id=np.array([i for i in range(len(model.ontology.classes()))])
    )

    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=img, detections=detections,
                                               labels=retrieve_labels(model.ontology.classes(), detections.class_id))

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    image = annotated_frame.astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask" + str(idx) + ".jpg"), image_rgb)

def plot_grid_masks(image, masks, classes, idx):
    # # Load the image and masks (replace with your own loading logic)
    # image = plt.imread('image.jpg')  # Shape: (H, W, 3)
    # masks = np.random.randint(0, 2, (58, image.shape[0], image.shape[1]))  # Example masks

    # Define a colormap for the masks
    cmap = ListedColormap(['none', 'red'])  # Transparent and Red

    # Create a grid of subplots
    rows, cols = 8, 8  # Adjust based on your desired layout
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    # Loop through each class and plot
    for i in range(58):
        # Overlay the mask onto the image
        #masked_image = image.copy()
        mask = masks[i]
        #masked_image[mask == 1] = [255, 0, 0]  # Example: mask overlay in red

        # Display in the corresponding grid cell
        axes[i].imshow(image)
        axes[i].imshow(mask, cmap=cmap, alpha=0.5)  # Overlay mask with transparency
        axes[i].axis('off')
        axes[i].set_title(f"{classes[i]}")

    # Turn off unused subplots
    for i in range(58, len(axes)):
        axes[i].axis('off')

    # Adjust layout and show the grid
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_masks" + str(idx) + ".jpg"))
    plt.close()

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

    def mask_generation(self, image):
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

    def mask_generation_with_all_concepts(self, image):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = self.model.predict(image)

        boxes = results.xyxy

        masks = results.mask

        classes = results.class_id

        confidence_score = results.confidence

        keep_indices = torch.ops.torchvision.nms(torch.tensor(boxes), torch.tensor(confidence_score), iou_threshold=0.9)
        # print(keep_indices)

        boxes = boxes[keep_indices]
        classes = classes[keep_indices]
        masks = masks[keep_indices]

        if boxes.ndim == 1:  # se dopo NMS resta una sola box
            boxes = np.expand_dims(boxes, axis=0)
            classes = np.array([classes])
            masks = np.expand_dims(masks, axis=0)

        masks, boxes = generate_mask_for_all_concepts(classes, masks, boxes)

        return boxes, masks, classes

    def automatic_mask_generation(self, image):

        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        sam2_checkpoint = "~/.cache/autodistill/segment_anything_2/sam2_hiera_large.pt"
        checkpoint = os.path.expanduser(sam2_checkpoint)
        model_cfg = "sam2_hiera_l.yaml"

        sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2)

        masks = mask_generator.generate(image)
        return masks


if __name__ == '__main__':
    caption = ("Airplane Wings/Cockpit windows/Engines/Runway/Clouds/Control tower/Headlights/Grille/Side "
               "mirrors/Wheels/Road/Traffic signs/Parking lot/Beak/Bird wings/Feather tail/Tree "
               "branches/Nest/Sky/Ears/Eyes/Tail/Whiskers/Sofa/Food bowl/Scratching post/Antlers/Short tail/Spotted "
               "fur/Forest/Grass/River/Collar/Tongue/Floppy ears/Leash/Park/Webbed feet/Warty "
               "skin/Pond/Rocks/Mane/Muzzle/Hooves/Fence/Saddle/Anchor/Flag/Hull/Sails/Sea/waves/Dock/Large "
               "wheels/Exhaust pipes/Cargo bed/Highway/Gas station")  #58 concepts
    #caption = " "

    #IMAGE_PATH = "images/flower.jpg"

    resize = 224

    train, val, test = load_classification_dataset("cifar10", "data", resize)

    dataset = ClassificationDataset(test)

    torch.cuda.empty_cache()

    model = GroundedSam2(caption, "Grounding DINO")

    # Mostra il path assoluto del file
    absolute_path = os.path.abspath(OUTPUT_DIR)
    print(f"Il file è stato salvato in: {absolute_path}")

    for idx in range(50,200):
        print("IMG " + str(idx))
        # Get the ground truth bounding boxes and labels
        image, ground_truth_labels = dataset.__getitem__(idx)

        image = ToPILImage()(image)

        if caption == " ":
            image = np.array(image)
            masks = model.automatic_mask_generation(image)
            save_automatic_generated_mask(masks, image)
        else:

            # Predict using the model
            boxes, masks, classes = model.mask_generation_with_all_concepts(image)

            #print(masks.shape)

            if len(masks) > 0:
                #save_images_with_mask(boxes,masks,classes,image,model,idx) #usable in case of printing only present class mask
                save_images_with_mask_for_all_concepts(image, masks, model, boxes)  #to print all the class masks, even if not present
                plot_grid_masks(image,masks,model.ontology.classes(),idx)

    # #bbox, categories, confidence = model(cv2.imread(IMAGE_PATH))
    #
    # image = Image.open(IMAGE_PATH)
    # image = np.array(image.convert("RGB"))
    #
    # # plt.figure(figsize=(20, 20))
    # # plt.imshow(image)
    # # plt.axis('off')
    # # plt.show()
    #
    # model.automatic_mask_generation(image)
    # # print(bbox)
    # # print(categories)
    # # print(confidence)
