import json

import cv2
import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms import ToPILImage

from src.utils import from_xywh_to_xyxy
import torchvision.transforms as transforms


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
        #print(type(image))
        # clip image to be three channels
        image = transforms.ToTensor()(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = ToPILImage()(image)
        #print(type(image))
        bboxes = elem['objects']['bbox']
        bboxes = [from_xywh_to_xyxy(bbox) for bbox in bboxes]  # conversion from xywh to xyxy
        categories = elem['objects'][self.name_for_category]
        # we are handling the dataset, and we have to map the labels with the indexes of the classes starting from 1
        categories = [cat-1 for cat in categories]
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
        self.classes = orig_dataset.info.features.to_dict()['objects']['feature']['label']['names'][1:]
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
