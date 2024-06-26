import hydra
import torch
import torchvision
import math
import pytorch_lightning as pl
import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

from src.utils import load_dataset, get_early_stopping, load_saliecy_method
from src.models.classifier import ClassifierModule
from src.datasets.dataset import SaliencyDataset
from src.log import get_loggers
from src.metrics import Insertion, Deletion, AverageDropIncrease


@hydra.main(config_path='config', config_name='config')
def main(cfg):

    # loggers = get_loggers(cfg)
    loggers = None

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
    model.eval()


    # load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    test = SaliencyDataset(test)
    dataloader =  torch.utils.data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    # load saliency method
    saliency_method = load_saliecy_method(cfg.saliency.method, model, device=cfg.train.device)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        logger=loggers,
    )

    #trainer.test(model, dataloader)

    device = torch.device(cfg.train.device)

    target_layer = cfg.target_layers[cfg.model.split('_Weights')[0]]

    model.eval()
    model_softmax = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))

    n_steps = cfg.metrics.n_steps

    insertion = Insertion(model_softmax, n_steps, cfg.train.batch_size, percentage=1., baseline=cfg.metrics.ins_obscure)
    deletion = Deletion(model_softmax, n_steps, cfg.train.batch_size, percentage=1., baseline=cfg.metrics.del_obscure)
    avg_drop_inc = AverageDropIncrease(model_softmax)

    model_softmax.eval()

    ins_auc_dict = dict()
    del_auc_dict = dict()

    for j, (images, labels) in enumerate(dataloader):
        #print('\n\nBatch:', j)
        saliency = saliency_method.generate_saliency(input_image=images, target_layer=target_layer).to(cfg.train.device)
        
        for i in range(images.shape[0]):
            image = images[i]
            image = image.to(cfg.train.device)
            saliency_map = saliency[i]
            saliency_map = saliency_map.to(cfg.train.device)
            labels = labels.to(cfg.train.device)
            ins_auc, ins_details = insertion(image, saliency, class_idx=labels[i])
            del_auc, del_details = deletion(image, saliency, class_idx=labels[i])
            #avgdrop, increase = avg_drop_inc(image, saliency_map, class_idx=labels[i])
            #print('Deletion - {:.5f}\nInsertion - {:.5f}'.format(del_auc, ins_auc))

            ins_auc_dict[labels[i].item()] = ins_auc
            del_auc_dict[labels[i].item()] = del_auc
        

        # remap ins_details and del_details on 100 points

        '''
        #for i in range(images.shape[0]):
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(images[i].permute(1, 2, 0))
            ax[1].imshow(images[i].permute(1, 2, 0))
            ax[1].imshow(saliency[i].cpu().detach().numpy(), cmap='jet', alpha=0.4)

            ax[2].plot(ins_details.cpu().numpy())
            # paint the area under the curve
            ax[2].fill_between(np.arange(ins_details.shape[0]), ins_details.cpu().numpy(), 0, alpha=0.5)
            #ax[2].set_ylim([0, 1])            
            ax[2].set_title('Insertion AUC: {:.5f}'.format(ins_auc_dict[labels[i].item()]))

            ax[3].plot(del_details.cpu().numpy())
            ax[3].fill_between(np.arange(del_details.shape[0]), del_details.cpu().numpy(), 0, alpha=0.5)
            #ax[3].set_ylim([0, 1])
            #ax[3].set_xlim([0, del_details.shape[0]])
            ax[3].set_title('Deletion AUC: {:.5f}'.format(del_auc_dict[labels[i].item()]))

            #plt.savefig('sample_{}.png'.format(i))
            plt.show()
        
        for i in range(images.shape[0]):
            # plot image + saliency map and saliency map
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(images[i].permute(1, 2, 0))
            ax[0].imshow(saliency[i].cpu().detach().numpy(), cmap='jet', alpha=0.3)
            ax[0].set_title('actual label: {}'.format(labels[i]))
            ax[1].imshow(saliency[i].cpu().detach().numpy(), cmap='jet')
            #pred = model_softmax(images[i].unsqueeze(0).to(cfg.train.device)).argmax().item()
            #ax[1].set_title('predicted label: {}'.format(pred))
            plt.show()
        '''
        
        if j == 5: # stop after 5 batches, just for testing
            break
        
    print('----------------------------------------------------------------')
    # move dict to cpu and print the mean
    ins_auc_dict = {k: v.cpu().numpy() for k, v in ins_auc_dict.items()}
    del_auc_dict = {k: v.cpu().numpy() for k, v in del_auc_dict.items()}
    print('Insertion AUC:', np.mean(list(ins_auc_dict.values())))
    print('Deletion AUC:', np.mean(list(del_auc_dict.values())))
    
        

    
    


if __name__ == '__main__':
    
    main()
