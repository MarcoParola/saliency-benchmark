import hydra
import os
import torch
from src.models.classifier import ClassifierModule
from pytorch_grad_cam import GradCAM
from torch import nn


class gradcam_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_image, target_class=None, target_layer=None):
        target_layers = get_layer_by_name(self.model, target_layer)

        if isinstance(target_layers, nn.Module):
            target_layers = [target_layers]

        # Ensure gradients are enabled and input_image requires grad
        input_image.requires_grad = True

        with torch.set_grad_enabled(True):
            cam = GradCAM(self.model, target_layers)
            saliency_maps = cam(input_image)

        saliency_tensor = torch.from_numpy(saliency_maps).detach()

        input_image.requires_grad = False

        return saliency_tensor


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in the model.")


# main for testing the interface of GradCAM made for this project
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torch.utils.data as data

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
        # model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Load the image
    images, _ = next(iter(dataloader))
    images = images.to('cpu')

    # Initialize the Saliency method
    target_layers_name = cfg.target_layers[cfg.model.split('_Weights')[0]]

    method = gradcam_interface(model, device=cfg.train.device, reshape_transform=False)
    saliency_maps = method.generate_saliency(input_image=images, target_layer=target_layers_name)

    for i in range(images.size(0)):
        image = images[i]
        saliency = saliency_maps[i]

        plt.figure(figsize=(5, 2.5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
