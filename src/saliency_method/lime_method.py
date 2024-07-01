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

    def generate_saliency(self, input_images, target_class=None, target_layer=None):
        # Move the images to the specified device
        images = input_images.to(self.device)

        # Convert images to numpy arrays
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert batch to numpy arrays

        # Initialize LimeImageExplainer
        explainer = LimeImageExplainer()

        # Initialize the list to store saliency maps for the batch
        batch_saliency_maps = []

        for image_np in images_np:
            # Generate explanations using the explainer for each image in the batch
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

            # Append the saliency map to the batch_saliency_maps list
            batch_saliency_maps.append(mask)

        saliency_maps = torch.tensor(batch_saliency_maps).to(self.device)

        return saliency_maps


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
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load the batch of images
    images, _ = next(iter(dataloader))
    images = images.to('cpu')

    # Initialize lime_interface
    lime = lime_interface(model, device='cpu')

    # Generate saliency maps for the batch of images
    saliency_maps = lime.generate_saliency(images)

    # Plot the saliency maps and the images
    for i in range(images.size(0)):
        image = images[i].permute(1, 2, 0).numpy()
        saliency_map = saliency_maps[i].squeeze().detach().numpy()

        plt.figure(figsize=(5, 2.5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(saliency_map, cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.show()
