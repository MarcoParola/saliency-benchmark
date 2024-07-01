import hydra
from torchvision import models
from tqdm import tqdm
import torch.nn as nn

from src.datasets.dataset import SaliencyDataset
from src.metrics_new import *
from src.models.classifier import ClassifierModule
from src.utils import *
from matplotlib import pyplot as plt

@hydra.main(config_path='config', config_name='config')
def main(cfg):
    # loggers = get_loggers(cfg)
    loggers = None

    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern.to(x.device), padding=klen // 2)

    '''
    # instantiate the model and load the weights
    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=cfg.train.finetune,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path)['state_dict'])
        # model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])
    model.eval()
    model_softmax = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))

    device = cfg.train.device
    model_softmax = model_softmax.to(device)

    model_softmax.eval()
    '''

    model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
    model = nn.Sequential(model, nn.Softmax(dim=1))

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path)['state_dict'])
        # model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    test = SaliencyDataset(test)
    dataloader = torch.utils.data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    # load saliency method
    saliency_method = load_saliecy_method(cfg.saliency.method, model, device=cfg.train.device)

    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    # Initialize metrics
    # step = 224  # Adjust step size as needed
    # substrate_fn = lambda x: torch.zeros_like(x)  # Function to replace pixels (e.g., with zeros)

    hw = int(cfg.dataset.resize) ** 2

    # Create CausalMetric instances for deletion and insertion
    deletion_metric = CausalMetric(model, mode='del', step=cfg.metrics.n_steps, substrate_fn=blur, dim=hw, n_classes=cfg[cfg.dataset.name].n_classes)
    insertion_metric = CausalMetric(model, mode='ins', step=cfg.metrics.n_steps, substrate_fn=torch.zeros_like, dim=hw, n_classes=cfg[cfg.dataset.name].n_classes)

    # Lists to store individual batch results
    scores = {'del': [], 'ins': []}

    for j, (images, labels) in enumerate(tqdm(dataloader, total=len(dataloader), desc='Loading images')):
        images = images.to(cfg.train.device)

        exp_batch = saliency_method.generate_saliency(input_images=images, target_layer=target_layer).to(
            cfg.train.device)

        del_auc, del_details = deletion_metric.evaluate(images, exp_batch,
                                                   cfg.train.batch_size)
        ins_auc, ins_details = insertion_metric.evaluate(images, exp_batch,
                                                     cfg.train.batch_size)

        # Calculate AUC for deletion and insertion and append to lists
        scores['del'].append(del_auc)
        scores['ins'].append(ins_auc)

        for i in range(images.shape[0]):
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(images[i].cpu().permute(1, 2, 0))  # Move tensor to CPU before converting to numpy
            ax[1].imshow(images[i].cpu().permute(1, 2, 0))  # Move tensor to CPU before converting to numpy
            ax[1].imshow(exp_batch[i].cpu().detach().numpy(), cmap='jet', alpha=0.4)

            ax[2].plot(ins_details[:, i])
            # paint the area under the curve
            ax[2].fill_between(np.arange(ins_details.shape[0]), ins_details[:, i], 0, alpha=0.5)
            # ax[2].set_ylim([0, 1])
            ax[2].set_title('Insertion AUC: {:.5f}'.format(scores['ins'][i]))

            ax[3].plot(del_details[:, i])
            ax[3].fill_between(np.arange(del_details.shape[0]), del_details[:, i], 0, alpha=0.5)
            # ax[3].set_ylim([0, 1])
            # ax[3].set_xlim([0, del_details.shape[0]])
            ax[3].set_title('Deletion AUC: {:.5f}'.format(scores['del'][i]))

            plt.savefig('sample_{}.png'.format(i))
            # plt.show()

        if j == 10:  # stop after 2 batches, just for testing
            break

    print('----------------------------------------------------------------')
    print('Final:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(np.mean(scores['del']), np.mean(scores['ins'])))


if __name__ == "__main__":
    main()
