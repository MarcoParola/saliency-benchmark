import torch
import torchvision
import numpy as np
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from supervision import ColorPalette
from torch import tensor
from torchvision import transforms

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


def save_annotated_images(label, image, results):
    print(label)
    print(results)
    box_annotator = sv.BoxAnnotator(color=ColorPalette(colors=[sv.Color(255,0,0)]))
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=results)

    label_annotator = sv.LabelAnnotator(color=ColorPalette(colors=[sv.Color(255,0,0)]))
    print(label)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=results, labels=label)

    # Assuming 'annotated_frame' is a PIL Image object
    if isinstance(annotated_frame, Image.Image):
        annotated_frame = np.array(annotated_frame)

    return annotated_frame

    #cv2.imwrite(os.path.join(OUTPUT_DIR, "image_box_" + str(iteration) + ".jpg"), annotated_frame)

    #mask_annotator = sv.MaskAnnotator()
    #annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=results)
    #cv2.imwrite(os.path.join(OUTPUT_DIR, "images_mask_" + str(iteration) + ".jpg"), annotated_frame)
    # label all images in a folder called `context_images`
    #model.label("../../../images", extension=".jpeg")


def save_annotated(image, true_boxes, ground_truth_labels, predicted_boxes, label_predicted, score_predicted,
                   label_pred_id, iteration):
    results = sv.Detections(
        xyxy=predicted_boxes,
        confidence=score_predicted,
        class_id=label_pred_id
    )
    image = save_annotated_images(label_predicted, image, results)

    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Create a plot
    plt.imshow(image_rgb)
    ax = plt.gca()

    # Draw predicted bounding boxes in blue with labels
    # for box,label in zip(predicted_boxes, label_predicted):
    #     x_min, y_min, x_max, y_max = box
    #
    #     rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='blue',
    #                          facecolor='none')
    #     ax.add_patch(rect)
    #     # Add the label text near the bounding box
    #     ax.text(x_min, y_min - 5, label, color='blue', fontsize=10, weight='bold')

    # Draw true bounding boxes in green with labels
    for box, label in zip(true_boxes, ground_truth_labels):
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='green',
                             facecolor='none')
        ax.add_patch(rect)
        # Add the label text near the bounding box
        ax.text(x_max, y_min - 5, label, color='green', fontsize=10, weight='bold')

    # Show the result
    output_path = os.path.join(OUTPUT_DIR, "image_box_" + str(iteration) + ".jpg")
    plt.axis('off')  # Turn off the axis
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()


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
