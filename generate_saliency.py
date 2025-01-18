import os

import cv2
import hydra
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

from src.datasets.classification import ClassificationDataset, load_classification_dataset
from src.models.classifier import ClassifierModule
from src.utils import *


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    print("Generate saliency")
    print(cfg.model)
    print(cfg.dataset.name)
    print(cfg.checkpoint)
    torch.cuda.empty_cache()
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

    # Initialize the Saliency method
    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    # load saliency method
    saliency_method = load_saliency_method(cfg.saliency.method, model, device=cfg.train.device)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    if cfg.saliency.dataset:
        print("FULL DATASET")
        # cfg.saliency.dataset == True, and so, we want to produce the saliency maps for all the dataset images

        # Load test dataset
        data_dir = os.path.join(cfg.mainDir, cfg.dataset.path)
        train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
        dataloader = data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

        if save_images:
            # Create directory to save saliency maps
            finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
            output_dir_images = os.path.join(cfg.mainDir, 'saliency_output/'+cfg.saliency.method+'_saliency_maps_images', finetune + cfg.model + cfg.dataset.name)
            output_dir_tensors = os.path.join(cfg.mainDir, 'saliency_output/'+cfg.saliency.method+'_saliency_maps_tensors', finetune + cfg.model + cfg.dataset.name)
            os.makedirs(output_dir_images, exist_ok=True)
            os.makedirs(output_dir_tensors, exist_ok=True)

        image_count = 0

        for images, labels in dataloader:
            if image_count >= 1:
                 break

            images = images.to(device)

            # Get model predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Generate saliency maps
            if cfg.saliency.method == "rise":
                saliency_maps = saliency_method.generate_saliency(images, preds)
            elif cfg.saliency.method == "lrp":
                saliency_maps = saliency_method.generate_saliency(input_images=images, batch=cfg.train.batch_size,resize=cfg.dataset.resize)
            else:
                saliency_maps = saliency_method.generate_saliency(input_images=images, target_layer=target_layer)

            for i in range(images.size(0)): #i-th images of the batch
                if image_count >= 1:
                     break
                print(image_count)

                image = images[i]
                image = image.cpu()
                saliency = saliency_maps[i]
                saliency = saliency.cpu()
                print("main")
                print(saliency.shape)
                predicted_class = preds[i]
                true_class = labels[i]

                # Create figure
                fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
                ax[0].imshow(image.squeeze().permute(1, 2, 0))
                ax[0].axis('off')
                ax[1].imshow(image.squeeze().permute(1, 2, 0))
                ax[1].imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
                ax[1].axis('off')
                ax[1].set_title(f'Pred: {predicted_class}\nTrue: {true_class}')

                if save_images:
                    if image_count<=200:
                        output_path = os.path.join(output_dir_images, f'saliency_map_{image_count}.png')
                        plt.savefig(output_path)
                    save_saliency_map(os.path.join(output_dir_tensors, f'saliency_map_{image_count}.pth'), saliency)
                else:
                    plt.show()

                plt.close(fig)

                image_count += 1
    else:
        # cfg.saliency.dataset == False, and so, we want to produce the saliency map only for one image, present in data/image
        print("SINGLE IMAGE")

        image = cv2.imread(f"data/image/{cfg.saliency.file_image}")

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((cfg.dataset.resize, cfg.dataset.resize)),
        ])

        img = transform(image)
        img = torch.clamp(img, 0, 1)  # clamp the image to be between 0 and 1
        # clip image to be three channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        image = img.unsqueeze(0)

        # Generate saliency maps
        if cfg.saliency.method == "rise":
            saliency_map = saliency_method.generate_saliency(image, [0])
        elif cfg.saliency.method == "lrp":
            saliency_map = saliency_method.generate_saliency(input_images=image, batch=cfg.train.batch_size,
                                                              resize=cfg.dataset.resize)
        else:
            saliency_map = saliency_method.generate_saliency(input_images=image, target_layer=target_layer)

        saliency = saliency_map.cpu()

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
        ax[0].imshow(image.squeeze().permute(1, 2, 0))
        ax[0].axis('off')
        if cfg.saliency.method != "lrp":
            ax[1].imshow(image.squeeze().permute(1, 2, 0))
        ax[1].imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
        ax[1].axis('off')
        #ax[1].set_title(f'Pred: {predicted_class}\nTrue: {true_class}')

        if save_images:
            output_dir = os.path.join(cfg.mainDir,'saliency_output', 'single_image_saliency')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f'{cfg.saliency.method}_saliency_map.png')
            plt.savefig(output_path)
            #save_saliency_map(os.path.join(output_dir_tensors, f'saliency_map_{image_count}.pth'), saliency)
        else:
            plt.show()

        plt.close(fig)


if __name__ == '__main__':
    main()