import hydra
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
import csv
from torch import nn

from pytorch_grad_cam import GradCAM

from src.datasets.classification import ClassificationDataset
from src.models.classifier import ClassifierModule
from src.utils import *

@hydra.main(config_path='../../config', config_name='config')
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
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataset = test
    dataloader = data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    if save_images:
        output_dir = os.path.join(cfg.currentDir, 'cifar100_output_dir')
        os.makedirs(output_dir, exist_ok=True)

        # Open CSV file for writing predicted and true classes
        csv_file_path = os.path.join(output_dir, 'predicted_vs_true_classes.csv')
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image_ID', 'True_Label', 'Predicted_Label'])  # Header of the CSV

    # Initialize the Saliency method
    target_layers_name = cfg.target_layers[cfg.model.split('_Weights')[0]]
    gradcam_method = gradcam_interface(model, device=cfg.train.device, reshape_transform=False)
    rise_method = rise_interface(model, device=cfg.train.device, input_size=(224, 224))
    lime_method = lime_interface(model, device=device)
    sidu_method = sidu_interface(model, device=cfg.train.device)

    image_count = 0
    class_tracker = set()  # To keep track of added classes

    # Prepare a single figure for all images
    fig, ax = plt.subplots(5, 5, figsize=(15, 15))  # 5 rows and 5 columns

    for images, labels in dataloader:
        if image_count >= 5:  # Only show 5 images
            break

        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Iterate through the batch of images
        for i in range(images.size(0)):
            if image_count >= 5:  # Only 5 unique classes
                break

            # Check if the prediction is correct and the class is not already added
            if preds[i] == labels[i] and labels[i].item() not in class_tracker:
                # Generate saliency maps
                gradcam_saliency_maps = gradcam_method.generate_saliency(input_images=images, target_layer=target_layers_name)
                rise_saliency_maps = rise_method.generate_saliency(images)
                sidu_saliency_maps = sidu_method.generate_saliency(images, target_layer=target_layers_name)
                lime_saliency_maps = lime_method.generate_saliency(images)

                image = images[i].cpu().squeeze().permute(1, 2, 0)
                gradcam_saliency = gradcam_saliency_maps[i].cpu().squeeze().detach().numpy()
                rise_saliency = rise_saliency_maps[i].cpu().squeeze().detach().numpy()
                sidu_saliency = sidu_saliency_maps[i].cpu().squeeze().detach().numpy()
                lime_saliency = lime_saliency_maps[i].cpu().squeeze().detach().numpy()

                # Add original image
                ax[image_count, 0].imshow(image)
                ax[image_count, 0].axis('off')
                if image_count == 0:
                    ax[image_count, 0].set_title("Original Image")

                # Add GradCAM
                ax[image_count, 1].imshow(image)
                ax[image_count, 1].imshow(gradcam_saliency, cmap='jet', alpha=0.4)
                ax[image_count, 1].axis('off')
                if image_count == 0:
                    ax[image_count, 1].set_title("GradCAM")

                # Add RISE
                ax[image_count, 2].imshow(image)
                ax[image_count, 2].imshow(rise_saliency, cmap='jet', alpha=0.4)
                ax[image_count, 2].axis('off')
                if image_count == 0:
                    ax[image_count, 2].set_title("RISE")

                # Add SIDU
                ax[image_count, 3].imshow(image)
                ax[image_count, 3].imshow(sidu_saliency, cmap='jet', alpha=0.4)
                ax[image_count, 3].axis('off')
                if image_count == 0:
                    ax[image_count, 3].set_title("SIDU")

                # Add LIME
                ax[image_count, 4].imshow(image)
                ax[image_count, 4].imshow(lime_saliency, cmap='jet', alpha=0.4)
                ax[image_count, 4].axis('off')
                if image_count == 0:
                    ax[image_count, 4].set_title("LIME")

                # Save the class to the set to avoid duplicates
                class_tracker.add(labels[i].item())

                # Save to CSV
                if save_images:
                    csv_writer.writerow([f"image_{image_count}", labels[i].item(), preds[i].item()])

                image_count += 1  # Increment the count of unique classes

    if save_images:
        output_path = os.path.join(output_dir, 'saliency_maps.png')
        plt.savefig(output_path)
        print(f'Image saved successfully: {output_path}')

        # Close the CSV file after writing
        csv_file.close()
        print(f'CSV saved successfully: {csv_file_path}')
    else:
        plt.show()

    plt.close(fig)

if __name__ == '__main__':
    main()
