from typing import List

import torch
import torchvision
import numpy as np
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from supervision import ColorPalette
from torch import tensor
from torchvision import transforms
from torchvision.ops import box_convert

import datasets
from src.datasets.classification import load_classification_dataset

from src.saliency_method.sidu import sidu_interface
from src.saliency_method.gradcam import gradcam_interface
from src.saliency_method.rise import rise_interface
from src.saliency_method.lime_method import lime_interface
import supervision as sv
import cv2
from PIL import Image
import matplotlib.pyplot as plt

OUTPUT_DIR = "..\..\..\..\output"


def from_xywh_to_xyxy(bbox):
    # the input is in format x,y,w,h and I want it in format x_top_left,y_top_left,x_bottom_right,y_bottom_right
    new_box = bbox.copy()
    new_box[2] = abs(bbox[0] + bbox[2])
    new_box[3] = abs(bbox[1] + bbox[3])
    return new_box


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


def load_saliecy_method(method, model, device='cpu', **kwargs):
    if method == 'sidu':
        return sidu_interface(model, device=device, **kwargs)
    elif method == 'gradcam':
        return gradcam_interface(model, device=device, **kwargs)
    elif method == 'rise':
        return rise_interface(model, device=device, **kwargs)
    elif method == 'lime':
        return lime_interface(model, device=device, **kwargs)
    else:
        raise ValueError(f'Unknown saliency method: {method}')


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
