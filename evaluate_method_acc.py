import hydra
import torch
import pytorch_lightning as pl
import os

from src.utils import load_dataset, get_early_stopping, load_saliecy_method
from src.models.classifier import ClassifierModule
from src.log import get_loggers

@hydra.main(config_path='config', config_name='config')
def main(cfg):

    # loggers = get_loggers(cfg)
    loggers = None

    # instantiate the model and load the weights
    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes, 
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )
    model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
    print(model_path)
    print(os.path.exists(model_path))
    print(torch.load(model_path).keys())
    model.load_state_dict(torch.load(model_path)['state_dict'])
    

    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataloader =  torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)

    saliency_method = load_saliecy_method(cfg.saliency_method, model, device=cfg.train.device)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        logger=loggers,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )

    #trainer.test(model, dataloader)

    from pytorch_sidu.utils.utils import load_torch_model_by_string
    model = load_torch_model_by_string('ResNet34_Weights.IMAGENET1K_V1')
    
    for image, _ in dataloader:
        saliency = saliency_method.generate_saliency(input_image=image, target_layer='model.layer4.1.conv2')
        
        # plot image + saliency map and saliency map
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image[0].permute(1, 2, 0))
        ax[0].imshow(saliency[0].cpu().detach().numpy(), cmap='jet', alpha=0.4)
        ax[1].imshow(saliency[0].cpu().detach().numpy(), cmap='jet')
        plt.show()

        # TODO compute partial accuracy of the saliency maps for this batch

    # TODO aggregate the average accuracy of the saliency maps for the whole dataset
    
    


if __name__ == '__main__':
    
    main()
