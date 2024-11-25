'''A quick example to generate heatmaps for vgg16.'''
import os
from functools import partial

import click
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg11, vgg11_bn, vgg16, vgg16_bn, resnet18, resnet50

from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import COMPOSITES
from zennit.core import Hook
from zennit.image import imsave, CMAPS
from zennit.layer import Sum
from zennit.torchvision import VGGCanonizer, ResNetCanonizer

import hydra

from src.datasets.classification import ClassificationDataset, load_classification_dataset
from src.models.classifier import ClassifierModule

from torchvision.models import vgg16_bn

from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.image import imsave

class lrp_interface:
    def __init__(self,model,device):
        self.model = model
        self.device = device

    def generate_saliency(self,input_images):

        saliency_maps = []
        for input_image in input_images:
            # Move the image to the specified device
            image = input_image.to(self.device)

            image = image.unsqueeze(0)

            # preprocess = transforms.Compose([
            #     transforms.Resize((224, 224)),  # Resize image to 224x224
            #     transforms.ToTensor(),  # Convert PIL image to Tensor
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means
            #         std=[0.229, 0.224, 0.225]  # Normalize with ImageNet stds
            #     )
            # ])
            #
            # # Load and preprocess the image
            # image = Image.open(image).convert('RGB')  # Ensure the image has 3 color channels
            # data = preprocess(image).unsqueeze(0)  # Add batch dimension

            # Create the one-hot tensor on the same device as the model
            target_tensor = torch.eye(self.model.num_classes, device=self.device)[[0]]

            canonizers = [SequentialMergeBatchNorm()]
            composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

            with Gradient(model=self.model, composite=composite) as attributor:
                out, relevance = attributor(image, target_tensor)

            heatmap = relevance.sum(1)   #sum over the color channels

            saliency_maps.append(heatmap)

        return saliency_maps
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    # Load the model and data
    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=cfg.train.finetune,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.mainDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load test dataset
    data_dir = os.path.join(cfg.mainDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataset = ClassificationDataset(test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    #if save_images:
    # Create directory to save saliency maps
    finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
    output_dir = os.path.join(cfg.mainDir, 'saliency_maps/lrp_saliency_maps', finetune + cfg.model + cfg.dataset.name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the Saliency method
    method = lrp_interface(model, device=cfg.train.device)

    image_count = 0

    for images, labels in dataloader:
        print(images.shape)
        if image_count >= 2:
            break

        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(input_images=images)

        for i in range(images.size(0)):
            if image_count >= 2:
                break

            image = images[i]   # here the image is a tensor

            # Convert to NumPy (H, W, C format for PIL)
            numpy_image = image.permute(1, 2, 0).cpu().numpy()

            # Normalize to 0-255 and convert to uint8
            numpy_image = (numpy_image * 255).astype('uint8')

            # Create a PIL image and save
            image = Image.fromarray(numpy_image)
            image.save(os.path.join(output_dir,"starting_image"+str(image_count)+".png"))

            saliency = saliency_maps[i]
            predicted_class = preds[i]
            true_class = labels[i]
            print("Relevance: ")
            print(saliency)

            amax = saliency.abs().cpu().numpy().max((1, 2))

            # save heat map with color map 'coldnhot'
            imsave(
                os.path.join(output_dir, f'saliency_map_{image_count}.png'),
                saliency[0].cpu(),
                vmin=-amax,
                vmax=amax,
                cmap='coldnhot',
                level=1.0,
                grid=False
            )

            # Open the saved heatmap image using Pillow
            heatmap_image = Image.open(os.path.join(output_dir, f'saliency_map_{image_count}.png')).convert("RGBA")

            # Add a title to the heatmap
            title_text = f"Pred: {predicted_class} | True: {true_class}"
            draw = ImageDraw.Draw(heatmap_image)

            # Choose a font and size (requires a font file)
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # Replace with a valid font path if needed
            except IOError:
                font = ImageFont.load_default()

            # Calculate text size and position using textbbox (new method)
            text_bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            image_width, image_height = heatmap_image.size
            text_position = ((image_width - text_width) // 2, 10)  # Centered at the top

            # Add the title text to the image
            draw.text(text_position, title_text, fill="white", font=font)

            heatmap_image.save(os.path.join(output_dir, f'saliency_map_{image_count}.png'))

            image_count += 1


if __name__ == '__main__':
    main()
    # # Define image preprocessing steps
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize image to 224x224
    #     transforms.ToTensor(),  # Convert PIL image to Tensor
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means
    #         std=[0.229, 0.224, 0.225]  # Normalize with ImageNet stds
    #     )
    # ])
    #
    # # Load and preprocess the image
    # image_path = '../../images/bird.jpg'  # Replace with your image file path
    # image = Image.open(image_path).convert('RGB')  # Ensure the image has 3 color channels
    # data = preprocess(image).unsqueeze(0)  # Add batch dimension
    # model = vgg16_bn()
    #
    # canonizers = [SequentialMergeBatchNorm()]
    # composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)
    #
    # with Gradient(model=model, composite=composite) as attributor:
    #     out, relevance = attributor(data, torch.eye(1000)[[0]])
    #
    # print(relevance)
    #
    # # sum over the color channels
    # heatmap = relevance.sum(1)
    # print(heatmap)
    # # get the absolute maximum, to center the heat map around 0
    # amax = heatmap.abs().numpy().max((1, 2))
    #
    # # save heat map with color map 'coldnhot'
    # imsave(
    #     'heatmap.png',
    #     heatmap[0],
    #     vmin=-amax,
    #     vmax=amax,
    #     cmap='coldnhot',
    #     level=1.0,
    #     grid=False
    # )

