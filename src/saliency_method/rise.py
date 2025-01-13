import hydra
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from src.datasets.classification import ClassificationDataset
from src.models.classifier import ClassifierModule
from src.utils import *


class RISE(nn.Module):
    def __init__(self, model, n_masks=500, p1=0.1, input_size=(224, 224), initial_mask_size=(7, 7), n_batch=32,
                 mask_path=None):
        super().__init__()
        self.model = model
        self.n_masks = n_masks
        self.p1 = p1
        self.input_size = input_size
        self.initial_mask_size = initial_mask_size
        self.n_batch = n_batch

        if mask_path is not None:
            self.masks = self.load_masks(mask_path)
        else:
            self.masks = self.generate_masks()

    def generate_masks(self):
        # cell size in the upsampled mask
        Ch = np.ceil(self.input_size[0] / self.initial_mask_size[0])
        Cw = np.ceil(self.input_size[1] / self.initial_mask_size[1])

        resize_h = int((self.initial_mask_size[0] + 1) * Ch)
        resize_w = int((self.initial_mask_size[1] + 1) * Cw)

        masks = []

        for _ in range(self.n_masks):
            # generate binary mask
            binary_mask = torch.randn(
                1, 1, self.initial_mask_size[0], self.initial_mask_size[1])
            binary_mask = (binary_mask < self.p1).float()

            # upsampling mask
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

            # random cropping
            i = np.random.randint(0, Ch)
            j = np.random.randint(0, Cw)
            mask = mask[:, :, i:i + self.input_size[0], j:j + self.input_size[1]]

            masks.append(mask)

        masks = torch.cat(masks, dim=0)  # (N_masks, 1, H, W)

        return masks

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def save_masks(self, filepath):
        torch.save(self.masks, filepath)

    def forward(self, x):
        # print("Shape of input image (x):", x.shape)
        # print("Shape of masks (self.masks):", self.masks.shape)
        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        # print("Shape of masked input (masked_x):", masked_x.shape)

        for i in range(0, self.n_masks, self.n_batch):
            input = masked_x[i:min(i + self.n_batch, self.n_masks)].to(device)
            # print("Shape of input to the model (input):", input.shape)
            out = self.model(input)
            probs.append(torch.softmax(out, dim=1).to('cpu').data)

        probs = torch.cat(probs)  # shape => (n_masks, n_classes)
        n_classes = probs.shape[1]

        # calculate saliency map using probability scores as weights
        saliency = torch.matmul(
            probs.data.transpose(0, 1),
            self.masks.view(self.n_masks, -1)
        )
        saliency = saliency.view(
            (n_classes, self.input_size[0], self.input_size[1]))
        saliency = saliency / (self.n_masks * self.p1)

        # normalize
        m, _ = torch.min(saliency.view(n_classes, -1), dim=1)
        saliency -= m.view(n_classes, 1, 1)
        M, _ = torch.max(saliency.view(n_classes, -1), dim=1)
        saliency /= M.view(n_classes, 1, 1)

        # print("Shape of saliency map (saliency):", saliency.shape)
        return saliency.data


class rise_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_images, predicted_class, target_class=None, target_layer=None):
        # Initialize the RISE method with the model
        rise_model = RISE(self.model)

        # Initialize the tensor to store saliency maps for the batch
        batch_saliencies = []

        i = 0

        # Generate the saliency maps for each image in the batch
        for image in input_images:
            # Generate the saliency map for the current image
            saliency_map = rise_model(image.unsqueeze(0).to(self.device))
            # print(saliency_map.shape)
            # print(saliency_map)
            saliency_map = saliency_map[predicted_class[i], :]  # Assuming saliency_map is of shape (1, num_channels, height, width)
            # print(saliency_map.shape)
            # print(saliency_map)

            # Append the saliency map to the batch_saliencies list
            batch_saliencies.append(saliency_map)
            i += 1

        # Stack the saliency maps along the batch dimension
        saliency_maps = torch.stack(batch_saliencies)

        return saliency_maps


# main for testing the interface of RISE made for this project
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    print("RISE")
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
    dataloader = data.DataLoader(test, batch_size=1, shuffle=True)

    # Flag to determine whether to save or show images
    save_images = cfg.visualize.save_images

    if save_images:
        # Create directory to save saliency maps
        finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
        output_dir_images = os.path.join(cfg.mainDir, 'saliency_output/RISE_saliency_maps_images',
                                         finetune + cfg.model + cfg.dataset.name)
        output_dir_tensors = os.path.join(cfg.mainDir, 'saliency_output/RISE_saliency_maps_tensors',
                                          finetune + cfg.model + cfg.dataset.name)
        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_tensors, exist_ok=True)

    # Initialize the Saliency method
    method = rise_interface(model, device=cfg.train.device, input_size=(224, 224))

    image_count = 0

    for images, labels in dataloader:
        # if image_count >= 5:
        #     break

        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(images, preds)
        #print(saliency_maps.shape)

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
