import hydra
import torch.utils.data as data
from src.utils import load_model, load_dataset, load_saliecy_method


@hydra.main(config_path='config', config_name='config')
def main(cfg):

    model = load_model(cfg.model, cfg.dataset.name)
    train, val, test = load_dataset(cfg.dataset)
    saliency_method = load_saliecy_method(cfg.saliency_method)

    dataloader = data.DataLoader(test, batch_size=1, shuffle=True)

    for image, _ in dataloader:
        saliency = saliency_method.generate_saliency(image, target_layer='layer4.2.conv2') # TODO change the target layer
        
        # TODO compute partial accuracy of the saliency maps for this batch

    # TODO aggregate the average accuracy of the saliency maps for the whole dataset


if __name__ == '__main__':
    main()