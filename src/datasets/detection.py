import json

import cv2
import torch


class DetectionDataset(torch.utils.data.Dataset):

    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform
        self.classes = orig_dataset.info.features.to_dict()['objects']['feature']['category']['names']

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        elem = self.orig_dataset[idx]
        image = elem['image']
        bbox = elem['objects']['bbox']
        categories = elem['objects']['category']
        return image, bbox, categories

    def __iter__(self):
        return self


if __name__ == "__main__":
    from datasets import load_dataset

    #ds = load_dataset("Francesco/animals-ij5d2", split="test")  #load the test set
    ds = load_dataset("rafaelpadilla/coco2017", split="val")
    dataset = DetectionDataset(ds)
    print(dataset.__len__())
    img, bbox, categories = dataset.__getitem__(0)
    img.show()
    print(bbox)
    print(categories)
    print(dataset.classes)