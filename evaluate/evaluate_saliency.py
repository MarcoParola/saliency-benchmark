import os
import hydra
import torch
from tqdm import tqdm
from src.datasets.classification import ClassificationDataset, load_classification_dataset
from src.models.classifier import ClassifierModule
from src.metrics.saliency_metrics import Insertion, Deletion
from src.utils import load_saliency_method


@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    loggers = None

    # instantiate the model and load the weights
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

    #qui sotto non è uguale a sopra?
    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        # model.load_state_dict(torch.load(model_path)['state_dict'])
        model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    test = ClassificationDataset(test)
    dataloader = torch.utils.data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    #qui dovremmo inserire la localization?
    insertion_metric = Insertion(model, n_pixels=cfg.metrics.n_pixels)
    deletion_metric = Deletion(model, n_pixels=cfg.metrics.n_pixels)

    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    # load saliency method
    saliency_method = load_saliency_method(cfg.saliency.method, model, device=cfg.train.device)

    # Lists to store AUC scores
    insertion_scores = []
    deletion_scores = []

    total_images = len(test)

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for j, (images, labels) in enumerate(dataloader):
            images = images.to(cfg.train.device)
            model = model.to(cfg.train.device)

            saliency = saliency_method.generate_saliency(input_images=images, target_layer=target_layer).to(
                cfg.train.device)

            for i in range(images.shape[0]):
                image = images[i]
                image = image.to(cfg.train.device)
                saliency_map = saliency[i]
                saliency_map.to(cfg.train.device)
                label = labels[i]
                label = label.to(cfg.train.device)

                image_to_mask = image.clone()  # Start with the original image

                # Compute the auc for Insertion and Deletion metric
                auc_ins_score = insertion_metric(image_to_mask, saliency_map, label, start_with_blurred=True)
                auc_del_score = deletion_metric(image_to_mask, saliency_map, label, start_with_blurred=False)

                # Append scores to lists
                insertion_scores.append(auc_ins_score)
                deletion_scores.append(auc_del_score)

                # Update progress bar
                pbar.update(1)
            '''
            if j == 2:
                break
            '''

    # Calculate and print the average AUC scores
    avg_auc_ins_score = sum(insertion_scores) / len(insertion_scores) if insertion_scores else 0
    avg_auc_del_score = sum(deletion_scores) / len(deletion_scores) if deletion_scores else 0

    # print(f'AUC Insertion Score: {avg_auc_ins_score}')
    # print(f'AUC Deletion Score: {avg_auc_del_score}')

    # Save results to file
    finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
    name_file = f"{finetune}_{cfg.model}_{cfg.dataset.name}_{cfg.saliency.method}.txt"
    output_file = os.path.join(cfg.currentDir, name_file)
    with open(output_file, 'w') as f:
        f.write(f'AUC Insertion Score: {avg_auc_ins_score}\n')
        f.write(f'AUC Deletion Score: {avg_auc_del_score}\n')


if __name__ == "__main__":
    main()
