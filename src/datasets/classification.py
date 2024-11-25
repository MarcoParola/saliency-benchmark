import torch
import numpy as np
import os
from torchvision import transforms
import torchvision
import datasets


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


if __name__ == "__main__":

    train, val, test = load_classification_dataset('cifar10', 'data', 224)
    dataset = ClassificationDataset(train)
    image, label = dataset.__getitem__(0)
    # print max and min values
    print(torch.max(image), torch.min(image))
    print(image.shape, label)
