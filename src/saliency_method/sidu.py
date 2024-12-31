import hydra
import matplotlib.pyplot as plt
import torch.utils.data as data

from pytorch_sidu import sidu

from src.datasets.classification import ClassificationDataset
from src.models.classifier import ClassifierModule
from src.utils import *


class sidu_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_images, target_class=None, target_layer=None):
        device = torch.device(self.device)
        saliency_maps = sidu(self.model, target_layer, input_images, device=device)
        return saliency_maps


# main for testing the interface of sidu made for this project
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

    # load test dataset
    data_dir = os.path.join(cfg.mainDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataset = ClassificationDataset(test)
    dataloader = data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    if save_images:
        # Create directory to save saliency maps
        finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
        output_dir_images = os.path.join(cfg.mainDir, 'saliency_output/SIDU_saliency_maps_images',
                                         finetune + cfg.model + cfg.dataset.name)
        output_dir_tensors = os.path.join(cfg.mainDir, 'saliency_output/SIDU_saliency_maps_tensors',
                                          finetune + cfg.model + cfg.dataset.name)
        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_tensors, exist_ok=True)

    # Initialize the Saliency method
    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]
    method = sidu_interface(model, device=cfg.train.device)

    image_count = 0

    for images, labels in dataloader:
        if image_count >= 5:
            break

        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(images, target_layer=target_layer)

        for i in range(images.size(0)):
            if image_count >= 5:
                break

            image = images[i].cpu()
            saliency = saliency_maps[i].cpu()
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
                output_path = os.path.join(output_dir_images, f'saliency_map_{image_count}.png')
                plt.savefig(output_path)
                save_saliency_map(os.path.join(output_dir_tensors, f'saliency_map_{image_count}.pth'), saliency)
            else:
                plt.show()

            plt.close(fig)

            image_count += 1


if __name__ == '__main__':
    main()
