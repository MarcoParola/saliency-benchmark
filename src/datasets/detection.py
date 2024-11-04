import json

import cv2
import torch
from datasets import load_dataset

from src.utils import from_xywh_to_xyxy


def load_detection_dataset(dataset_name):
    dataset = None

    if dataset_name == 'francesco_animals':
        ds = load_dataset("Francesco/animals-ij5d2", split="test")  # load the test set
        dataset = AnimalDetectionDataset(ds)

    if dataset_name == 'coco2017':
        ds = load_dataset("rafaelpadilla/coco2017", split="val")
        dataset = CocoDetectionDataset(ds)

    return dataset


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        elem = self.orig_dataset.__getitem__(idx)
        image = elem['image']
        bboxes = elem['objects']['bbox']
        bboxes = [from_xywh_to_xyxy(bbox) for bbox in bboxes]  # conversion from xywh to xyxy
        categories = elem['objects'][self.name_for_category]
        #categories = [self.classes[cat] for cat in categories] # in this way I have the text of the label and not the number
        return image, bboxes, categories

    def __iter__(self):
        return self


class AnimalDetectionDataset(DetectionDataset):
    def __init__(self, orig_dataset, transform=None):
        super().__init__(orig_dataset)
        self.classes = orig_dataset.info.features.to_dict()['objects']['feature']['category']['names'][1:11]
        self.name_for_category = 'category'

class CocoDetectionDataset(DetectionDataset):
    def __init__(self, orig_dataset, transform=None):
        super().__init__(orig_dataset)
        self.classes = orig_dataset.info.features.to_dict()['objects']['feature']['label']['names']
        self.name_for_category = 'label'


if __name__ == "__main__":

    dataset = load_detection_dataset("francesco_animals")
    print(dataset.__len__())
    print(dataset.classes)
    for i in range(dataset.__len__()):
        img, bbox, categories = dataset.__getitem__(i)
        # test su shape
        print(img.size)
        print(img.mode)
