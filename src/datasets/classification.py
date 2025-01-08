import cv2
import torch
import numpy as np
import os

from PIL import Image
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torchvision import transforms
import torchvision
import kagglehub
import datasets
import json


def get_images_intel_images(directory):
    dataset = []
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    label = 0

    for labels in os.listdir(directory):  # Main Directory where each class label is present as folder name.
        if labels == 'glacier':  # Folder contain Glacier Images get the '2' class label.
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3

        for image_file in os.listdir(
                directory + labels):  # Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory + labels + r'/' + image_file)  # Reading the image (OpenCV)
            image = cv2.resize(image, (
                150, 150))  # Resize the image, Some images are different sizes. (Resizing is very Important)
            dataset.append({"image": image, "label": label})

    #     return Images, Labels
    return shuffle(dataset, random_state=42), ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def get_classlabel_intel_images(class_code):
    labels = {2: 'glacier', 4: 'sea', 0: 'buildings', 1: 'forest', 5: 'street', 3: 'mountain'}

    return labels[class_code]


def load_classification_dataset(dataset, data_dir, resize=256, val_split=0.2, test_split=0.2):
    train, val, test = None, None, None

    torch.manual_seed(42)
    np.random.seed(42)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((resize, resize)),
    ])

    # CIFAR-10
    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # CIFAR-100
    elif dataset == 'cifar100':
        train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # Caltech101
    elif dataset == 'caltech101':
        data = torchvision.datasets.Caltech101(data_dir, download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split + test_split:], indices[:val_split], indices[
                                                                                              val_split:val_split + test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)


    # ImageNet
    elif dataset == 'imagenet':
        imsize = 299

        preprocess = transforms.Compose([
            transforms.Resize((imsize, imsize)),  # 이미지의 크기를 변경
            transforms.ToTensor(),  # torch.Tensor 형식으로 변경 [0, 255] → [0, 1]
        ])

        data_dir = '../../data/ILSVRC2012_img_val_subset'
        val = torchvision.datasets.ImageFolder(os.path.join(data_dir), preprocess)
        train = val
        test = val
        '''
        from torchvision.datasets import ImageFolder
        val = datasets.load_dataset('mrm8488/ImageNet1K-val', split='train')

        class ImageNetDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, lbl = self.dataset[idx]['image'], self.dataset[idx]['label']
                if self.transform:
                    img = self.transform(img)
                # check if the image contains 1 channels
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                return img, lbl

        val = ImageNetDataset(val, transform)
        train = val
        test = val
    '''

    # Oxford-IIIT Pet
    elif dataset == 'oxford-iiit-pet':
        data = torchvision.datasets.OxfordIIITPet(data_dir, download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split + test_split:], indices[:val_split], indices[
                                                                                              val_split:val_split + test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)

    # Oxford Flowers
    elif dataset == 'oxford-flowers':
        data = torchvision.datasets.Flowers102(data_dir, split='train', download=True, transform=transform)
        val_data = torchvision.datasets.Flowers102(data_dir, split='val', download=True, transform=transform)
        test_data = torchvision.datasets.Flowers102(data_dir, split='test', download=True, transform=transform)

        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        train = data
        val = val_data
        test = test_data

    # SVHN
    elif dataset == 'svhn':
        # data = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transform)
        train = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=transform)
        test = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # MNIST
    elif dataset == 'mnist':
        train = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # FashionMNIST
    elif dataset == 'fashionmnist':
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    elif dataset == 'imagenette':
        train = load_dataset('frgfm/imagenette', 'full_size', split='train')
        classes = train.info.features.to_dict()['label']['names']
        test = load_dataset('frgfm/imagenette', 'full_size', split='validation')
        train = train.shuffle(seed=42)
        test = test.shuffle(seed=42)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])
        train = IntelImagenetteDataset(train, classes, transform)
        test = IntelImagenetteDataset(test, classes, transform)
        val = IntelImagenetteDataset(val, classes, transform)

    elif dataset == 'intel_image':
        # Download latest version
        path = kagglehub.dataset_download("puneet6060/intel-image-classification")

        train_path = os.path.join(path, "seg_train/seg_train/")
        pred_path = os.path.join(path, "seg_pred")
        test_path = os.path.join(path, "seg_test/seg_test/")

        train, classes = get_images_intel_images(train_path)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])
        test, classes = get_images_intel_images(test_path)

        train = IntelImagenetteDataset(train, classes, transform)
        val = IntelImagenetteDataset(val, classes, transform)
        test = IntelImagenetteDataset(test, classes, transform)

    elif dataset == 'oral':
        train = OralClassificationDataset("data/oral/train.json", transform)
        val = OralClassificationDataset("data/oral/val.json", transform)
        test = OralClassificationDataset("data/oral/test.json", transform)

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return train, val, test


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        img, lbl = self.orig_dataset.__getitem__(idx)
        img = torch.clamp(img, 0, 1)  # clamp the image to be between 0 and 1
        # clip image to be three channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)

        return img, lbl


class IntelImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, classes, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        elem = self.orig_dataset.__getitem__(idx)
        image = elem['image']
        lbl = elem['label']
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        image = torch.clamp(image, 0, 1)  # clamp the image to be between 0 and 1

        # clip image to be three channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, lbl


class OralClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None):
        self.annonations = annonations
        self.transform = transform

        with open(annonations, "r") as f:
            self.dataset = json.load(f)

        self.images = dict()
        for image in self.dataset["images"]:
            self.images[image["id"]] = image

        self.categories = dict()
        for i, category in enumerate(self.dataset["categories"]):
            self.categories[category["id"]] = i

    def __len__(self):
        return len(self.dataset["annotations"])

    def __getitem__(self, idx):
        annotation = self.dataset["annotations"][idx]
        image = self.images[annotation["image_id"]]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1","oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        category = self.categories[annotation["category_id"]]

        return image, category


if __name__ == "__main__":
    train, val, test = load_classification_dataset('intel_image', 'data', 224)
    print(test.classes)
    print(len(train))
    print(len(val))
    print(len(test))
    image, label = train.__getitem__(0)
    # Convert (C, H, W) to (H, W, C) for Matplotlib
    image = image.permute(1, 2, 0).cpu().numpy()

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    print(image.shape, label)
    image, label = val.__getitem__(0)
    # Convert (C, H, W) to (H, W, C) for Matplotlib
    image = image.permute(1, 2, 0).cpu().numpy()

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    print(image.shape, label)
    image, label = test.__getitem__(0)
    # Convert (C, H, W) to (H, W, C) for Matplotlib
    image = image.permute(1, 2, 0).cpu().numpy()

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    # print max and min values
    # print(torch.max(image), torch.min(image))
    # print(image.shape, label)
