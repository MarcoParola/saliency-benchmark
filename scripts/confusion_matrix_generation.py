import itertools

import hydra
import numpy as np
import torch
import pytorch_lightning as pl
import os

from src.utils import get_early_stopping, get_save_model_callback
from src.models.classifier import ClassifierModule
from src.datasets.classification import ClassificationDataset, load_classification_dataset
from src.log import get_loggers
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def compute_cm(predictions, labels, classes, model, dataset_name):
    # Concatenare tutte le previsioni e le etichette
    all_predictions = np.concatenate(predictions, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    # Calcolare la confusion matrix
    cm = confusion_matrix(all_predictions, all_labels)

    # Stampa o salva la confusion matrix
    print("Confusion Matrix:\n", cm)

    plt.imshow(cm, interpolation='nearest', cmap='OrRd')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    print(tick_marks)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    print("Normalized confusion matrix")
    thresh = 0.6

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    output_dir = os.path.abspath('confusion_matrices')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join("confusion_matrices", "cm_" + model + "_" + dataset_name + ".jpg"))
    plt.close()


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    # Set seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)

    # callback
    callbacks = list()
    callbacks.append(get_early_stopping(cfg.train.patience))
    finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
    model_save_dir = os.path.join(cfg.currentDir, "checkpoints", finetune + cfg.model + cfg.dataset.name)
    callbacks.append(get_save_model_callback(model_save_dir))

    # loggers
    loggers = get_loggers(cfg)

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_classification_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)

    print(test.classes)

    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=cfg.train.batch_size,
                                              shuffle=False,
                                              num_workers=cfg.train.num_workers)

    model = ClassifierModule(
        weights=cfg.model,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=cfg.train.finetune,
        lr=cfg.train.lr,
        max_epochs=cfg.train.max_epochs
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        logger=loggers,
        callbacks=callbacks,
        #deterministic=True
    )

    if cfg.dataset.name != 'imagenet':
        model_path = os.path.join(cfg.currentDir, cfg.checkpoint)
        model.load_state_dict(torch.load(model_path)['state_dict'])
        # model.load_state_dict(torch.load(model_path, map_location=cfg.train.device)['state_dict'])

    model.eval()

    trainer.test(model, test_loader)

    compute_cm(model.all_predictions, model.all_labels, test.classes, cfg.model, cfg.dataset.name)


if __name__ == '__main__':
    main()
