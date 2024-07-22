import hydra
import os
from pytorch_sidu import sidu
import torch
from src.models.classifier import ClassifierModule


class sidu_interface():
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
    from torchvision.models import resnet34, ResNet34_Weights
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torch.utils.data as data

    # Load the model and data
    # model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
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
    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    method = sidu_interface(model, device=cfg.train.device)
    saliency_maps = method.generate_saliency(images, target_layer=target_layer)

    # Plot the saliency map and the image for each image in the batch
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
