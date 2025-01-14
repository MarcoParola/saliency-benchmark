import csv
import glob
from typing import List

import pandas as pd
import torch
import torchvision
import numpy as np
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from supervision import ColorPalette
from torch import tensor
from torchvision import transforms
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage

import datasets
from src.datasets.classification import load_classification_dataset

from src.saliency_method.sidu import sidu_interface
from src.saliency_method.gradcam import gradcam_interface
from src.saliency_method.rise import rise_interface
from src.saliency_method.lime_method import lime_interface
import supervision as sv
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

OUTPUT_DIR = "..\..\..\..\output"


def get_dataset_classes(annotation_dir):
    classes = set()

    for filename in os.listdir(annotation_dir):
        if not filename.endswith(".xml"):
            continue
        filepath = os.path.join(annotation_dir, filename)

        tree = ET.parse(filepath)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            classes.add(class_name)

    return sorted(classes)


def handle_dataset_kaggle(images_dir, annotations_dir):
    # Verify directories exist
    if not os.path.isdir(images_dir) or not os.path.isdir(annotations_dir):
        raise FileNotFoundError("Expected directories JPEGImages or Annotations not found in the dataset path")

    # Retrieve all image-annotation pairs
    dataset = []

    # Collect image paths and corresponding annotation file paths
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    annotation_files = glob.glob(os.path.join(annotations_dir, "*.xml"))

    # Create a dictionary of annotation files by their base name for easy lookup
    annotations_dict = {os.path.splitext(os.path.basename(ann))[0]: ann for ann in annotation_files}

    # Process each image file and retrieve bounding boxes
    for image_file in image_files:
        # Extract the file name without extension to find the corresponding XML annotation
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        annotation_file = annotations_dict.get(image_id)

        if annotation_file is None:
            #print(f"No annotation file for {image_id}")
            continue

        # Parse the XML annotation file
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        # List of bounding boxes for this image
        bboxes = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text

            # Get bounding box coordinates
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Append bounding box and class to the list
            bboxes.append({
                "class": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

        # Store the image path and bounding boxes in the dataset
        dataset.append({
            "image_path": image_file,
            "bboxes": bboxes
        })

    # Example of printing the image and bbox data
    print("Number of images processed:", len(dataset))
    print("Sample entry:")
    print(dataset[0])

    classes = get_dataset_classes(annotations_dir)
    print(classes)
    return dataset, classes


def from_xywh_to_xyxy(bbox):
    # the input is in format x,y,w,h and I want it in format x_top_left,y_top_left,x_bottom_right,y_bottom_right
    new_box = bbox.copy()
    new_box[2] = abs(bbox[0] + bbox[2])
    new_box[3] = abs(bbox[1] + bbox[3])
    return new_box

def resize_boxes(bbox, scale_x, scale_y):
    # Scale the bounding box coordinates
    bbox[0] *= scale_x  # Scale x_min and x_max
    bbox[2] *= scale_x
    bbox[1] *= scale_y  # Scale y_min and y_max
    bbox[3] *= scale_y
    return bbox


def from_array_to_dict(boxes_vector, labels_vector):
    # print(boxes_vector)
    # print(labels_vector)
    # Convert lists to tensors
    boxes_tensor = torch.tensor(boxes_vector)
    labels_tensor = torch.tensor(labels_vector)

    # Create the structure
    structure = [
        {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }
    ]

    #print(structure)

    return structure


def from_array_to_dict_predicted(boxes_vector, confidence_scores, labels_vector):
    boxes_vector = tensor(boxes_vector)
    confidence_scores = tensor(confidence_scores)
    labels_vector = tensor(labels_vector)

    structure = [dict(
        boxes=boxes_vector,
        scores=confidence_scores,
        labels=labels_vector
    )]
    return structure


def retrieve_labels(param, class_id):
    labels = []
    for cl in class_id:
        labels.append(param[cl])
    return labels


def save_annotated_images(label, image, results, position):
    image = ToPILImage()(image)
    box_annotator = sv.BoxAnnotator(color=ColorPalette(colors=[sv.Color(255, 0, 0)]))
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=results)

    label_annotator = sv.LabelAnnotator(color=ColorPalette(colors=[sv.Color(255, 0, 0)]), text_position=position)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=results, labels=label)

    # Assuming 'annotated_frame' is a PIL Image object
    if isinstance(annotated_frame, Image.Image):
        annotated_frame = np.array(annotated_frame)

    return annotated_frame


def save_annotated(image, true_boxes, ground_truth_labels, predicted_boxes, label_predicted, score_predicted,
                   label_pred_id, label_true_id, iteration):
    if len(predicted_boxes) > 0:
        #draw predicted bounding box
        results = sv.Detections(
            xyxy=np.array(predicted_boxes),
            confidence=np.array(score_predicted),
            class_id=np.array(label_pred_id)
        )

        image = save_annotated_images(label_predicted, image, results, sv.Position.TOP_LEFT)

    if len(true_boxes) > 0:
        #draw true bounding box
        results = sv.Detections(
            xyxy=np.array(true_boxes),
            class_id=np.array(label_true_id)
        )

        image = save_annotated_images(ground_truth_labels, image, results, sv.Position.TOP_RIGHT)

    image = image.astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    cv2.imwrite(os.path.join(OUTPUT_DIR, "image_box" + str(iteration) + ".jpg"), image_rgb)

def from_normalized_cxcywh_to_xyxy(image, boxes):
    h, w, _ = image.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return np.array(xyxy)

def draw_grounding_dino_prediction(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor,
                                   phrases: List[str]) -> np.ndarray:
    # h, w, _ = image_source.shape
    # boxes = boxes * torch.Tensor([w, h, w, h])
    # print("Final boxes:"+str(boxes))
    #xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    #print(xyxy)
    #print(boxes)
    detections = sv.Detections(xyxy=np.array(boxes), class_id=np.arange(len(boxes)))

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator(color=ColorPalette(colors=[sv.Color(0, 255, 0)]))
    #annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=image_source, detections=detections)
    label_annotator = sv.LabelAnnotator(color=ColorPalette(colors=[sv.Color(0, 255, 0)]),
                                        text_position=sv.Position.BOTTOM_LEFT)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def save_annotated_grounding_dino(image, true_boxes, ground_truth_labels, predicted_boxes, label_predicted,
                                  score_predicted,
                                  label_pred_id, label_true_id, iteration):
    if len(predicted_boxes) > 0:
        #draw predicted bounding box
        image = draw_grounding_dino_prediction(np.array(image), torch.Tensor(predicted_boxes), score_predicted, label_predicted)

    if len(true_boxes) > 0:
        #draw true bounding box
        results = sv.Detections(
            xyxy=np.array(true_boxes),
            class_id=np.array(label_true_id)
        )

        image = save_annotated_images(ground_truth_labels, image, results, sv.Position.TOP_RIGHT)

    image = np.array(image)
    image = image.astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    cv2.imwrite(os.path.join(OUTPUT_DIR, "image_box" + str(iteration) + ".jpg"), image_rgb)


def get_save_model_callback(save_path):
    """Returns a ModelCheckpoint callback
    cfg: hydra config
    """
    save_model_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=save_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
    )
    return save_model_callback


def get_early_stopping(patience=10):
    """Returns an EarlyStopping callback
    cfg: hydra config
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
    )
    return early_stopping_callback


def load_saliency_method(method, model, device='cpu', **kwargs):
    if method == 'sidu':
        return sidu_interface(model, device=device, **kwargs)
    elif method == 'gradcam':
        return gradcam_interface(model, device=device, **kwargs)
    elif method == 'rise':
        return rise_interface(model, device=device, **kwargs)
    elif method == 'lime':
        return lime_interface(model, device=device, **kwargs)
    elif method == 'lrp':
        return  lrp_interface(model,)
    else:
        raise ValueError(f'Unknown saliency method: {method}')

def save_saliency_map(save_path, saliency_map):
    torch.save(saliency_map.to(torch.float16),save_path) # took to float16 to consume less memory

def load_saliency_map(input_path):
    saliency = torch.load(input_path)
    return saliency.to(torch.float32) # took back to float32

def save_mask(save_path, mask):
    torch.save(torch.from_numpy(mask), save_path)

def load_mask(input_path):
    tensor_loaded = torch.load(input_path)
    return tensor_loaded.to(torch.float32)

def save_list(path_file, list):
    with open(path_file,'w') as f:
        f.write('\n'.join(list))

def load_list(path_file):
    with open(path_file, 'r') as f:
        list = f.read().splitlines()
    return list

def retrieve_concepts(dataset_name):
    # Initialize an empty list to store concepts
    all_concepts = []

    absolute_path = os.path.abspath("concepts")

    # Read the CSV file
    with open(os.path.join(absolute_path,dataset_name+"_concepts.csv"), mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(row)
            # Split the concepts in the current row and extend the list
            concepts = row["concepts"].split(";")
            all_concepts.extend(concepts)

    # Format the concepts as a single string separated by "/"
    caption = "/".join(all_concepts)
    #num_concepts = len(all_concepts)

    print("Caption:", caption)
    return caption


if __name__ == "__main__":

    data = [
        # 'cifar10',
        # 'cifar100',
        # 'caltech101',
        'oxford-flowers',
        # 'imagenet',
        # 'oxford-iiit-pet',
        # 'svhn',
        # 'mnist',
        # 'fashionmnist',
    ]
    for dataset in data:
        print(f'\n\nDataset: {dataset}')
        data = load_classification_dataset(dataset, './data')
        print(data[0].__len__(), data[1].__len__(), data[2].__len__())

        test = data[2]
        print(test)
        import matplotlib.pyplot as plt

        for i in range(10):
            img, lbl = test.__getitem__(i)
            print(img.shape, lbl)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(lbl)
            plt.show()

        # import matplotlib.pyplot as plt
        # dataloader = torch.utils.data.DataLoader(data[2], batch_size=4, shuffle=True)
        # for i, batch in enumerate(dataloader):
        #     for j in range(4):
        #         img, lbl = batch[0][j], batch[1][j]
        #         print(img.shape, lbl)
        #         plt.imshow(img.permute(1, 2, 0))
        #         plt.title(lbl)
        #         plt.show()

        '''
        for d in data:
            print(f'\nData: {d}')

            for i in range(len(d)):
                _, label = d[i]
                if label not in class_distribution:
                    class_distribution[label] = 0
                class_distribution[label] += 1
            
            # sort and print the class distribution
            class_distribution = dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))
            # for key, value in class_distribution.items():
            #     print(f'{key}: {value}')

            # print number of classes
            print(f'Number of classes: {len(class_distribution)}')

            # compute the class unbalance as the ratio between the number of samples in the most frequent class and the number of samples in the least frequent class
            dist = list(class_distribution.values())
            class_unbalance = max(dist) / min(dist)
            print(f'Class unbalance: {class_unbalance}')
        '''
