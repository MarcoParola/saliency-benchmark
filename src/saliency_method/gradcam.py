import torch
from pytorch_grad_cam import GradCAM
from torch import nn
import cv2


class gradcam_interface():
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_image, target_class=None, target_layer=None):
        target_layers = get_layer_by_name(self.model, target_layer)

        if isinstance(target_layers, nn.Module):
            target_layers = [target_layers]

        cam = GradCAM(self.model, target_layers)
        saliency_maps = cam(input_image)
        saliency_tensor = torch.from_numpy(saliency_maps)
        return saliency_tensor


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in the model.")


# main for testing the interface of sidu made for this project
if __name__ == '__main__':
    from torchvision.models import resnet34, ResNet34_Weights
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torch.utils.data as data

    # Load the model and data
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Load the image
    image, _ = next(iter(dataloader))
    image = image.to('cpu')

    # Initialize the Saliency method
    target_layers_name = 'layer4.2.conv2'

    method = gradcam_interface(model, device='cpu', reshape_transform=None)
    saliency = method.generate_saliency(input_image=image, target_layer=target_layers_name)

    # plot the saliency map and the image
    plt.figure(figsize=(5, 2.5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.imshow(saliency[0].squeeze().detach().numpy(), cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.show()