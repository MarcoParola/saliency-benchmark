import hydra
import os
from src.models.classifier import ClassifierModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RISE(nn.Module):
    def __init__(self, model, n_masks=50, p1=0.1, input_size=(224, 224), initial_mask_size=(7, 7), n_batch=128, mask_path=None):
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
            mask = mask[:, :, i:i+self.input_size[0], j:j+self.input_size[1]]

            masks.append(mask)

        masks = torch.cat(masks, dim=0)   # (N_masks, 1, H, W)

        return masks

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def save_masks(self, filepath):
        torch.save(self.masks, filepath)

    def forward(self, x):
        #print("Shape of input image (x):", x.shape)
        #print("Shape of masks (self.masks):", self.masks.shape)
        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        #print("Shape of masked input (masked_x):", masked_x.shape)

        for i in range(0, self.n_masks, self.n_batch):
            input = masked_x[i:min(i + self.n_batch, self.n_masks)].to(device)
            #print("Shape of input to the model (input):", input.shape)
            out = self.model(input)
            probs.append(torch.softmax(out, dim=1).to('cpu').data)

        probs = torch.cat(probs)    # shape => (n_masks, n_classes)
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

        #print("Shape of saliency map (saliency):", saliency.shape)
        return saliency.data



class rise_interface():
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    '''
    def generate_saliency(self, input_image, target_class=None, target_layer=None):
        # Initialize the RISE method with the model
        rise_model = RISE(self.model)

        # Generate the saliency map using the RISE method
        saliency_maps = rise_model(input_image)
        saliency_maps = saliency_maps[0, :]
        return saliency_maps
    '''

    def generate_saliency(self, input_images, target_class=None, target_layer=None):
        # Initialize the RISE method with the model
        rise_model = RISE(self.model)

        # Initialize the tensor to store saliency maps for the batch
        batch_saliencies = []

        # Generate the saliency maps for each image in the batch
        for image in input_images:
            # Generate the saliency map for the current image
            saliency_map = rise_model(image.unsqueeze(0).to(self.device))
            saliency_map = saliency_map[0, :]  # Assuming saliency_map is of shape (1, num_channels, height, width)

            # Append the saliency map to the batch_saliencies list
            batch_saliencies.append(saliency_map)

        # Stack the saliency maps along the batch dimension
        saliency_maps = torch.stack(batch_saliencies)

        return saliency_maps

# main for testing the interface of RISE made for this project
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
    method = rise_interface(model, device=cfg.train.device, input_size=(224, 224))
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