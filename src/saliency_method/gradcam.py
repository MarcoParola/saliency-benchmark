import hydra
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch import nn

from pytorch_grad_cam import GradCAM

from src.datasets.dataset import SaliencyDataset
from src.models.classifier import ClassifierModule
from src.utils import *


class gradcam_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_images, target_class=None, target_layer=None):
        target_layers = get_layer_by_name(self.model, target_layer)

        if isinstance(target_layers, nn.Module):
            target_layers = [target_layers]

        # Ensure gradients are enabled and input_image requires grad
        input_images.requires_grad = True

        with torch.set_grad_enabled(True):
            cam = GradCAM(self.model, target_layers)
            saliency_maps = cam(input_images)

        saliency_tensor = torch.from_numpy(saliency_maps).detach()

        input_images.requires_grad = False

        return saliency_tensor


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in the model.")


# main for testing the interface of GradCAM made for this project
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
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataset = SaliencyDataset(test)
    dataloader = data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    if save_images:
        # Create directory to save saliency maps
        finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
        output_dir = os.path.join(cfg.mainDir, 'gradCam_saliency_maps', finetune + cfg.model + cfg.dataset.name)
        os.makedirs(output_dir, exist_ok=True)

    # Initialize the Saliency method
    target_layers_name = cfg.target_layers[cfg.model.split('_Weights')[0]]
    method = gradcam_interface(model, device=cfg.train.device, reshape_transform=False)

    image_count = 0

    for images, labels in dataloader:
        if image_count >= 2:
            break

        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(input_images=images, target_layer=target_layers_name)

        for i in range(images.size(0)):
            if image_count >= 2:
                break

            image = images[i]
            saliency = saliency_maps[i]
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
                output_path = os.path.join(output_dir, f'saliency_map_{image_count}.png')
                plt.savefig(output_path)
            else:
                plt.show()

            plt.close(fig)

            image_count += 1


if __name__ == '__main__':
    main()
