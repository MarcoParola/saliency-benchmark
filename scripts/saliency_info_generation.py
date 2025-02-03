import os

import cv2
import hydra
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

from src.datasets.classification import ClassificationDataset, load_classification_dataset
from src.models.classifier import ClassifierModule
from src.utils import *


# def create_csv_file(dataset, output_csv):
#     """
#        Processes a dataset and generates a CSV with id, true class, predicted class, and saliency map path.
#
#        Parameters:
#        - dataset: iterable containing the dataset information
#        - output_csv: path to the output CSV file
#        """
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write the header
#         writer.writerow(["ID", "True Class", "Predicted Class", "GRADCAM path", "LIME path", "SIDU path", "LRP path"])
#
#         dataloader = data.DataLoader(test, batch_size=16, shuffle=True)
#
#         # Loop through the dataset and write to CSV
#         for idx in range(dataset.__len__()):
#             print("IMG " + str(idx))
#             # Get the ground truth bounding boxes and labels
#             image, ground_truth_labels = dataset.__getitem__(idx)
#             true_class = data.get("true_class", "unknown")
#             predicted_class = data.get("predicted_class", "unknown")
#             saliency_path = data.get("saliency_path", "unknown")
#             writer.writerow([idx, true_class, predicted_class, saliency_path])


@hydra.main(config_path='../config', config_name='config', version_base=None)
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
    dataloader = data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    image_count = 0

    output_dir_csv = os.path.join(cfg.mainDir, 'saliency_info', cfg.model)

    if not os.path.exists(output_dir_csv):
        os.makedirs(output_dir_csv)

    output_csv = os.path.join(output_dir_csv, f'saliency_info_{cfg.dataset.name}.csv')

    # Create for the specified model a csv file containing id_image,true_class,pred_class,saliency_tensor for
    # each method

    # Retrieve the path for the tensors of each saliency method
    finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
    gradcam_path = os.path.join(cfg.mainDir, 'saliency_output/gradcam_saliency_maps_tensors',
                                finetune + cfg.model + cfg.dataset.name)
    rise_path = os.path.join(cfg.mainDir, 'saliency_output/rise_saliency_maps_tensors',
                             finetune + cfg.model + cfg.dataset.name)
    lrp_path = os.path.join(cfg.mainDir, 'saliency_output/lrp_saliency_maps_tensors',
                            finetune + cfg.model + cfg.dataset.name)
    sidu_path = os.path.join(cfg.mainDir, 'saliency_output/sidu_saliency_maps_tensors',
                             finetune + cfg.model + cfg.dataset.name)
    lime_path = os.path.join(cfg.mainDir, 'saliency_output/lime_saliency_maps_tensors',
                             finetune + cfg.model + cfg.dataset.name)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["ID", "True Class", "Predicted Class", "gradcam path", "lime path","rise path", "sidu path"])

        for images, labels in dataloader:

            images = images.to(device)

            # Get model predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):  #i-th images of the batch

                print(image_count)

                predicted_class = preds[i]
                true_class = labels[i]

                writer.writerow([image_count, true_class, predicted_class,
                                 os.path.join(gradcam_path, f"saliency_map_{image_count}.pth"),
                                 os.path.join(lime_path, f"saliency_map_{image_count}.pth"),
                                 # os.path.join(lrp_path, f"saliency_map_{image_count}.pth"),
                                 os.path.join(rise_path, f"saliency_map_{image_count}.pth"),
                                 os.path.join(sidu_path, f"saliency_map_{image_count}.pth")])
                image_count += 1


if __name__ == '__main__':
    main()
