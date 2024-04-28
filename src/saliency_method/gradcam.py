
# TODO import gradcam

class gradcam_interface():
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_image, target_class=None, target_layer=None):
        saliency_maps = # TODO call gradcam
        return saliency_maps



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
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Load the image
    image, _ = next(iter(dataloader))
    image = image.to('cpu')

    # Initialize the Saliency method
    target_layer = 'layer4.2.conv2'

    method = gradcam_interface(model, device='cpu')
    saliency = method.generate_saliency(...) # TODO complete the arguments

    # plot the saliency map and the image
    plt.figure(figsize=(5, 2.5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.show()