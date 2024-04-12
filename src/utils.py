import torch
import torchvision
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import timm
import detectors
import numpy as np


def load_model(model_name, dataset):
    model = None

    # CIFAR-10
    if dataset == 'cifar10':
        if model_name == 'resnet18':
            model = timm.create_model("resnet18_cifar10", pretrained=True)
        elif model_name == 'resnet34':
            model = timm.create_model("resnet34_cifar10", pretrained=True)
        elif model_name == 'resnet50':
            model = timm.create_model("resnet50_cifar10", pretrained=True)
        elif model_name == 'vgg16':
            model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
        elif model_name == 'vgg19':
            pass # TODO load model trained by us
        elif model_name == 'convnext-b':
            model = timm.create_model("hf-hub:anonauthors/cifar10-timm-convnext_base.fb_in1k", pretrained=True)
        elif model_name == 'convnext-t':
            pass # TODO load model trained by us
        elif model_name == 'densenet121':
            model = timm.create_model("edadaltocg/densenet121_cifar10", pretrained=True)
        elif model_name == 'densenet169':
            pass # TODO load model trained by us
        elif model_name == 'vit-b':
            model = timm.create_model("hf-hub:anonauthors/cifar10-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)


    # CIFAR-100
    elif dataset == 'cifar100':
        
        if model_name == 'resnet18':
            model = timm.create_model("resnet18_cifar100", pretrained=True)
        elif model_name == 'resnet34':
            model = timm.create_model("resnet34_cifar100", pretrained=True)
        elif model_name == 'resnet50':
            model = timm.create_model("resnet50_cifar100", pretrained=True)
        elif model_name == 'vgg16':
            model = timm.create_model("vgg16_bn_cifar100", pretrained=True)
        elif model_name == 'vgg19':
            pass # TODO load model trained by us
        elif model_name == 'convnext-b':
            model = timm.create_model("hf-hub:anonauthors/cifar100-timm-convnext_base.fb_in1k", pretrained=True)
        elif model_name == 'convnext-t':
            pass # TODO load model trained by us
        elif model_name == 'densenet121':
            model = timm.create_model("densenet121_cifar100", pretrained=True)
        elif model_name == 'densenet169':
            pass # TODO load model trained by us
        elif model_name == 'vit-b':
            model = timm.create_model("hf-hub:anonauthors/cifar100-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    # Caltech101
    elif dataset == 'caltech101':
        if model_name == 'resnet18':
            pass # TODO load model trained by us
        elif model_name == 'resnet34':
            pass # TODO load model trained by us
        elif model_name == 'resnet50':
            model = timm.create_model('hf-hub:anonauthors/caltech101-timm-resnet50', pretrained=True)
        elif model_name == 'vgg16':
            pass # TODO load model trained by us
        elif model_name == 'vgg19':
            pass # TODO load model trained by us
        elif model_name == 'convnext-b':
            model = timm.create_model('hf-hub:anonauthors/caltech101-timm-convnext_base.fb_in1k', pretrained=True)
        elif model_name == 'convnext-t':
            pass # TODO load model trained by us
        elif model_name == 'densenet121':
            pass # TODO load model trained by us
        elif model_name == 'densenet169':
            pass # TODO load model trained by us
        elif model_name == 'vit-b':
            model = timm.create_model("hf-hub:anonauthors/caltech101-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    # ImageNet
    elif dataset == 'imagenet':
        if model_name == 'resnet18':
            model = sidu.load_torch_model_by_string('ResNet18_Weights.IMAGENET1K_V1')
        elif model_name == 'resnet34':
            model = sidu.load_torch_model_by_string('ResNet34_Weights.IMAGENET1K_V1')
        elif model_name == 'resnet50':
            model = sidu.load_torch_model_by_string('ResNet50_Weights.IMAGENET1K_V1')
        elif model_name == 'vgg16':
            model = sidu.load_torch_model_by_string('VGG16_Weights.IMAGENET1K_V1')
        elif model_name == 'vgg19':
            model = sidu.load_torch_model_by_string('VGG19_Weights.IMAGENET1K_V1')
        elif model_name == 'convnext-b':
            model = sidu.load_torch_model_by_string('ConvNeXt_Base_Weights.IMAGENET1K_V1')
        elif model_name == 'convnext-t':
            model = sidu.load_torch_model_by_string('ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        elif model_name == 'densenet121':
            model = sidu.load_torch_model_by_string('DenseNet121_Weights.IMAGENET1K_V1')
        elif model_name == 'densenet169':
            model = sidu.load_torch_model_by_string('DenseNet169_Weights.IMAGENET1K_V1')
        elif model_name == 'vit-b':
            model = sidu.load_torch_model_by_string('ViT_B_16_Weights.IMAGENET1K_V1')


    # Oxford-IIIT Pet
    elif dataset == 'oxford-iiit-pet':
        if model_name == 'resnet18':
            pass # TODO load model trained by us
        elif model_name == 'resnet34':
            pass # TODO load model trained by us
        elif model_name == 'resnet50':
            model = timm.create_model('hf-hub:nateraw/resnet50-oxford-iiit-pet',pretrained=True)
        elif model_name == 'vgg16':
            pass # TODO load model trained by us
        elif model_name == 'vgg19':
            pass # TODO load model trained by us
        elif model_name == 'convnext-b':
            model = timm.create_model("hf-hub:anonauthors/oxford_pet-timm-convnext_base.fb_in1k", pretrained=True)
        elif model_name == 'convnext-t':
            pass # TODO load model trained by us
        elif model_name == 'densenet121':
            pass # TODO load model trained by us
        elif model_name == 'densenet169':
            pass # TODO load model trained by us
        elif model_name == 'vit-b':
            model = timm.create_model("hf-hub:anonauthors/oxford_pet-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    # SVHN
    elif dataset == 'svhn':
        if model_name == 'resnet18':
            model = timm.create_model("resnet18_svhn", pretrained=True)
        elif model_name == 'resnet34':
            model = timm.create_model("resnet34_svhn", pretrained=True)
        elif model_name == 'resnet50':
            model = timm.create_model("resnet50_svhn", pretrained=True)
        elif model_name == 'vgg16':
            model = timm.create_model("vgg16_bn_svhn", pretrained=True)
        elif model_name == 'vgg19':
            pass # TODO load model trained by us
        elif model_name == 'convnext-b':
            pass # TODO load model trained by us
        elif model_name == 'convnext-t':
            pass # TODO load model trained by us
        elif model_name == 'densenet121':
            model = timm.create_model("densenet121_svhn", pretrained=True)
        elif model_name == 'densenet169':
            pass # TODO load model trained by us
        elif model_name == 'vit-b':
            model = timm.create_model("vit_base_patch16_224_in21k_ft_svhn", pretrained=True)

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return model

def load_dataset(dataset, val_split=0.2, test_split=0.2):

    train, val, test = None, None, None

    torch.manual_seed(42)
    np.random.seed(42)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
    ])

    # CIFAR-10
    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # CIFAR-100
    elif dataset == 'cifar100':
        train = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # Caltech101
    elif dataset == 'caltech101':
        data = torchvision.datasets.Caltech101('./data', download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split+test_split:], indices[:val_split], indices[val_split:val_split+test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)
    # ImageNet
    elif dataset == 'imagenet':
        from torchvision.datasets import ImageFolder
        test = ImageFolder(root='./data/imagenet/val', transform=transform)

    # Oxford-IIIT Pet
    elif dataset == 'oxford-iiit-pet':
        data = torchvision.datasets.OxfordIIITPet('./data', download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split+test_split:], indices[:val_split], indices[val_split:val_split+test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)

    # SVHN
    elif dataset == 'svhn':
        #data = torchvision.datasets.SVHN('./data', split='test', download=True, transform=transform)
        train = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transform)
        test = torchvision.datasets.SVHN('./data', split='test', download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return train, val, test

    
if __name__ == "__main__":

    datasets = ['cifar10', 'cifar100', 'caltech101', 'oxford-iiit-pet', 'svhn']
    for dataset in datasets:
        print(f'\n\nDataset: {dataset}')
        data = load_dataset(dataset)
        print(data[0].__len__(), data[1].__len__(), data[2].__len__())
        class_distribution = {}
        for d in data:
            print(f'\nData: {d}')
            for i in range(len(d)):
                _, label = d[i]
                if label not in class_distribution:
                    class_distribution[label] = 0
                class_distribution[label] += 1
            
            # sort and print the class distribution
            class_distribution = dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))
            for key, value in class_distribution.items():
                print(f'{key}: {value}')