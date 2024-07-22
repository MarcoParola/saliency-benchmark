import hydra
import os

from torchvision.models import resnet34, ResNet34_Weights

from src.models.classifier import ClassifierModule
from lime.lime_image import LimeImageExplainer
import torch


class lime_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model.to(device)
        self.device = device
        self.kwargs = kwargs

    def batch_predict(self, images):
        # Set the model to evaluation mode
        self.model.eval()

        # Convert the list of images to a batch tensor
        images = torch.stack([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in images], dim=0).to(
            self.device)

        with torch.no_grad():
            # Get the model's predictions
            logits = self.model(images)

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def generate_saliency(self, input_images, target_class=None, target_layer=None):
        # Initialize LimeImageExplainer
        explainer = LimeImageExplainer()

        # Iterate over each image in the batch
        saliency_maps = []
        for input_image in input_images:
            # Move the image to the specified device
            image = input_image.to(self.device)

            # Convert image to numpy array
            image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()

            # Generate explanations using the explainer
            explanation = explainer.explain_instance(
                image_np,
                self.batch_predict,
                top_labels=1 if target_class is None else target_class,
                hide_color=None,
                num_samples=100
            )

            # Get the target class if not specified
            if target_class is None:
                target_class_value = explanation.top_labels[0]

            # Get the image and mask for the specified class
            _, mask = explanation.get_image_and_mask(
                target_class_value,
                positive_only=False,
                num_features=15,
                hide_rest=False
            )

            #saliency_maps.append(mask)
            saliency_maps.append(torch.tensor(mask, dtype=torch.float32))

        saliency_maps = torch.stack(saliency_maps)

        return saliency_maps


# main for testing the interface of lime made for this project
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
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

    # Load the images
    images, _ = next(iter(dataloader))
    images = images.to('cpu')

    # Initialize the Saliency method
    method = lime_interface(model, device='cpu')
    saliency_maps = method.generate_saliency(images)

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
