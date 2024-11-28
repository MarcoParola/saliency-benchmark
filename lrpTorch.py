import hydra
from PIL import Image
from torchvision.transforms import transforms

from lrp import Sequential
from src.datasets.classification import load_classification_dataset, ClassificationDataset
from src.models.classifier import ClassifierModule

import torch
import lrp


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    # Load the model and data
    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=cfg.train.finetune,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means
            std=[0.229, 0.224, 0.225]  # Normalize with ImageNet stds
        )
    ])

    # Load and preprocess the image
    image_path = 'images/bird.jpg'  # Replace with your image file path
    image = Image.open(image_path).convert('RGB')  # Ensure the image has 3 color channels
    data = preprocess(image).unsqueeze(0)  # Add batch dimension
    y_hat = model.forward(data)
    y_hat = y_hat[torch.arange(cfg.train.batch_size), y_hat.max(1)[1]]  # Choose maximizing output neuron
    y_hat = y_hat.sum()

    # Backward pass (do explanation)
    y_hat.backward()
    explanation = data.grad
    print(explanation)


if __name__ == '__main__':
    #main()

    import os
    import sys
    import torch
    import pickle
    from torch.nn import Sequential, Conv2d, Linear

    import pathlib
    import argparse
    import torchvision
    from torchvision import datasets, transforms as T
    import configparser

    import numpy as np
    import matplotlib.pyplot as plt

    # Append parent directory of this file to sys.path,
    # no matter where it is run from
    base_path = pathlib.Path(__file__).parent.parent.absolute()
    sys.path.insert(0, base_path.as_posix())

    import lrp
    from lrp.patterns import fit_patternnet, fit_patternnet_positive  # PatternNet patterns


    def store_patterns(file_name, patterns):
        with open(file_name, 'wb') as f:
            pickle.dump([p.detach().cpu().numpy() for p in patterns], f)


    def load_patterns(file_name):
        with open(file_name, 'rb') as f: p = pickle.load(f)
        return p

    def project(X, output_range=(0, 1)):
        absmax = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
        X /= absmax + (absmax == 0).astype(float)
        X = (X + 1) / 2.  # range [0, 1]
        X = output_range[0] + X * (output_range[1] - output_range[0])  # range [x, y]
        return X


    def heatmap(X, cmap_name="seismic"):
        cmap = plt.cm.get_cmap(cmap_name)

        if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()

        shape = X.shape
        tmp = X.sum(axis=-1)  # Reduce channel axis

        tmp = project(tmp, output_range=(0, 255)).astype(int)
        tmp = cmap(tmp.flatten())[:, :3].T
        tmp = tmp.T

        shape = list(shape)
        shape[-1] = 3
        return tmp.reshape(shape).astype(np.float32)


    def clip_quantile(X, quantile=1):
        """Clip the values of X into the given quantile."""
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
        if not isinstance(quantile, (list, tuple)):
            quantile = (quantile, 100 - quantile)

        low = np.percentile(X, quantile[0])
        high = np.percentile(X, quantile[1])
        X[X < low] = low
        X[X > high] = high

        return X


    def grid(a, nrow=3, fill_value=1.):
        bs, h, w, c = a.shape

        # Reshape to grid
        rows = bs // nrow + int(bs % nrow != 0)
        missing = (nrow - bs % nrow) % nrow
        if missing > 0:  # Fill empty spaces in the plot
            a = np.concatenate([a, np.ones((missing, h, w, c)) * fill_value], axis=0)

        # Border around images
        a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.5)
        a = a.reshape(rows, nrow, h + 2, w + 2, c)
        a = np.transpose(a, (0, 2, 1, 3, 4))
        a = a.reshape(rows * (h + 2), nrow * (w + 2), c)
        return a


    def heatmap_grid(a, nrow=3, fill_value=1., cmap_name="seismic", heatmap_fn=heatmap):
        # Compute colors
        a = heatmap_fn(a, cmap_name=cmap_name)
        return grid(a, nrow, fill_value)

    torch.manual_seed(1337)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Normalization as expected by pytorch vgg models
    # https://pytorch.org/docs/stable/torchvision/models.html
    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
    _std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))


    def unnormalize(x):
        return x * _std + _mean


    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=_mean.flatten(),
                    std=_std.flatten()),
    ])

    train, val, test = load_classification_dataset("mnist","./data",224)
    dataset = ClassificationDataset(test)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # # # # # VGG model
    vgg_num = int(sys.argv[1]) if len(sys.argv) > 1 else 16  # Default to vgg16

    vgg = getattr(torchvision.models, "vgg%i" % vgg_num)(pretrained=True).to(device)
    # vgg = torchvision.models.vgg16(pretrained=True).to(device)
    vgg.eval()

    print("Loaded vgg-%i" % vgg_num)

    lrp_vgg = lrp.convert_vgg(vgg).to(device)
    # # # # #

    # Check that the vgg and lrp_vgg models does the same thing
    for x, y in train_loader: break
    x = x.to(device)
    x.requires_grad_(True)

    y_hat = vgg(x)
    y_hat_lrp = lrp_vgg.forward(x)

    assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), "\n\n%s\n%s\n%s" % (
    str(y_hat.view(-1)[:10]), str(y_hat_lrp.view(-1)[:10]), str((torch.abs(y_hat - y_hat_lrp)).max()))
    print("Done testing")
    # # # # #

    # # # # # # Patterns for PatternNet and PatternAttribution
    # patterns_path = (base_path / 'examples' / 'patterns' / ('vgg%i_pattern_pos.pkl' % vgg_num)).as_posix()
    # if not os.path.exists(patterns_path):
    #     patterns = fit_patternnet_positive(lrp_vgg, train_loader, device=device)
    #     store_patterns(patterns_path, patterns)
    # else:
    #     patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]
    #
    # print("Loaded patterns")


    # # # # # Plotting
    def compute_and_plot_explanation(rule, ax_, patterns=None, plt_fn=heatmap_grid):
        # Forward pass
        y_hat_lrp = lrp_vgg.forward(x, explain=True, rule=rule, pattern=patterns)

        # Choose argmax
        y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
        y_hat_lrp = y_hat_lrp.sum()

        # Backward pass (compute explanation)
        y_hat_lrp.backward()
        attr = x.grad

        # Plot
        attr = plt_fn(attr)
        ax_.imshow(attr)
        ax_.set_title(rule)
        ax_.axis('off')


    # PatternNet is typically handled a bit different, when visualized.
    def signal_fn(X):
        if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
        X = clip_quantile(X)
        X = project(X)
        X = grid(X)
        return X


    explanations = [
        # rule                  Pattern     plt_fn          Fig. pos
        ('alpha1beta0', None, heatmap_grid, (1, 0)),
        ('epsilon', None, heatmap_grid, (0, 1)),
        ('gamma+epsilon', None, heatmap_grid, (1, 1)),
        # ('patternnet', patterns, signal_fn, (0, 2)),
        # ('patternattribution', patterns, heatmap_grid, (1, 2)),
    ]

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    print("Plotting")

    # Plot inputs
    input_to_plot = unnormalize(x).permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
    input_to_plot = grid(input_to_plot, 3, 1.)
    ax[0, 0].imshow(input_to_plot)
    ax[0, 0].set_title("Input")
    ax[0, 0].axis('off')

    # Plot explanations
    for i, (rule, pattern, fn, (p, q)) in enumerate(explanations):
        compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)

    fig.tight_layout()
    fig.savefig((base_path /  ("vgg%i_explanations.png" % vgg_num)).as_posix(), dpi=280)
    plt.show()