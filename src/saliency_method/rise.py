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
        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        for i in range(0, self.n_masks, self.n_batch):
            input = masked_x[i:min(i + self.n_batch, self.n_masks)].to(device)
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
        return saliency.data



class rise_interface():
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_image, target_class=None, target_layer=None):
        # Initialize the RISE method with the model
        rise_model = RISE(self.model)

        # Generate the saliency map using the RISE method
        saliency_maps = rise_model(input_image)
        saliency_maps = saliency_maps[0, :]
        return saliency_maps


# main for testing the interface of RISE made for this project
if __name__ == '__main__':
    from torchvision.models import resnet34
    import torchvision.transforms as transforms
    import torch.utils.data as data
    import torchvision.datasets as datasets
    import matplotlib.pyplot as plt

    # Carica il modello e i dati
    model = resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Inizializza l'interfaccia RISE
    method = rise_interface(model, device='cpu', input_size=(224, 224))

    for i, (images, _) in enumerate(dataloader):

        images = images.to('cpu')

        # Genera le mappe di salienza per tutte le immagini nel batch
        saliencies = [method.generate_saliency(image.unsqueeze(0)) for image in images]

        # Visualizza le mappe di salienza e le immagini originali
        for j in range(images.size(0)):
            image = images[j]
            saliency = saliencies[j]

            plt.figure(figsize=(5, 2.5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze().permute(1, 2, 0))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(image.squeeze().permute(1, 2, 0))
            plt.imshow(saliency.squeeze().detach().numpy(), cmap='jet', alpha=0.4)
            plt.axis('off')
            plt.show()

        if i == 10:
            break
