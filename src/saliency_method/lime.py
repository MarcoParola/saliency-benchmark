from lime.lime_image import LimeImageExplainer
import torch

class lime_interface():
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

    def generate_saliency(self, input_image, target_class=None, target_layer=None):
        # Move the image to the specified device
        image = input_image.to(self.device)

        # Convert image to numpy array
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()

        # Initialize LimeImageExplainer
        explainer = LimeImageExplainer()

        # Generate explanations using the explainer
        explanation = explainer.explain_instance(
            image_np,
            self.batch_predict,
            top_labels=1 if target_class is None else [target_class],
            hide_color=None,
            num_samples=100
        )

        # Get the target class if not specified
        if target_class is None:
            target_class = explanation.top_labels[0]

        # Get the image and mask for the specified class
        _, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=False,
            num_features=15,
            hide_rest=False
        )

        return mask # TODO forse lo chiamerei saliency_map come nelle altre interfacce



# main for testing the interface of sidu made for this project
if __name__ == "__main__":

    import torch
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Load the model and data
    model = models.resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load the image
    image, _ = next(iter(dataloader))
    image = image.to('cpu')

    # Initialize lime_interface
    lime_interface = lime_interface(model, device='cpu')

    # Generate saliency map
    saliency_map = lime_interface.generate_saliency(image)

    # Plot the saliency map and the image
    plt.figure(figsize=(5, 2.5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.imshow(saliency_map, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.show()