import torch
import torchvision
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pytorch_lightning import LightningModule
import os

class ClassifierModule(LightningModule):

    def __init__(self, weights, num_classes, lr=10e-6, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        
        self.num_classes = num_classes
        self.backbone, self.classifier, self.preprocess = self.return_backbone_and_classifier(weights)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        self.backbone.eval()
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        imgs, labels = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)
        # compute metrics
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
        return accuracy


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, label = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        imgs, labels = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        loss = self.loss(y_hat, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss

    def return_backbone_and_classifier(self, model_name):
        backbone, classifier, preproc = None, None, None

        weights_cls = model_name.split(".")[0]
        weights_name = model_name.split(".")[1]
        name = model_name.split("_Weights")[0].lower()
        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)
        model = getattr(torchvision.models, name)(weights=weights)

        weights_cls = str(weights_cls)
        if 'ResNet' in model_name:
            backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
            classifier = torch.nn.Linear(model.fc.in_features, self.num_classes)

        elif 'EfficientNet' in model_name:
            backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
            classifier = torch.nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif 'VGG' in model_name:
            backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
            classifier = torch.nn.Linear(model.classifier[0].in_features, self.num_classes)

        return backbone, classifier, weights.transforms()


    
if __name__ == '__main__':
    
    model_list = [
        'ResNet50_Weights.IMAGENET1K_V1',
        'EfficientNet_B1_Weights.IMAGENET1K_V1',
        'VGG16_Weights.IMAGENET1K_V1',    
    ]

    img = torch.randn(2, 3, 256, 256)

    for model_name in model_list:
        print(model_name)
        model = ClassifierModule(model_name, 10)
        print(model(img).shape)

