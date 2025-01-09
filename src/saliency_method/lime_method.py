import hydra
import matplotlib.pyplot as plt
import torch.utils.data as data

from lime.lime_image import LimeImageExplainer

from src.datasets.classification import ClassificationDataset
from src.models.classifier import ClassifierModule
from src.utils import *


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

            saliency_maps.append(torch.tensor(mask, dtype=torch.float32))

        saliency_maps = torch.stack(saliency_maps)

        return saliency_maps


# main for testing the interface of lime made for this project
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    print("LIME")
    print(cfg.model)
    print(cfg.dataset.name)
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
        output_dir_images = os.path.join(cfg.mainDir, 'saliency_output/LIME_saliency_maps_images',
                                         finetune + cfg.model + cfg.dataset.name)
        output_dir_tensors = os.path.join(cfg.mainDir, 'saliency_output/LIME_saliency_maps_tensors',
                                          finetune + cfg.model + cfg.dataset.name)
        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_tensors, exist_ok=True)

    # Initialize the Saliency method
    method = lime_interface(model, device=device)

    image_count = 0

    for images, labels in dataloader:
        # if image_count >= 5:
        #     break
        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(images)

        for i in range(images.size(0)):
            # if image_count >= 5:
            #     break
            print(image_count)

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
                if image_count <= 200:
                    output_path = os.path.join(output_dir_images, f'saliency_map_{image_count}.png')
                    plt.savefig(output_path)
                save_saliency_map(os.path.join(output_dir_tensors, f'saliency_map_{image_count}.pth'), saliency)
            else:
                plt.show()

            plt.close(fig)

            image_count += 1


if __name__ == '__main__':
    main()
