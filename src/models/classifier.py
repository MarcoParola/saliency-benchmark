import torch
import torchvision
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pytorch_lightning import LightningModule
import os

class ClassifierModule(LightningModule):

    def __init__(self, weights, num_classes, finetune=False, lr=10e-6, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        self.model_name = weights.split("_Weights")[0].lower()
        self.num_classes = num_classes
        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)
        self.model = getattr(torchvision.models, self.model_name)(weights=weights)

        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False
            
        # method to set the classifier head independently of the model (as head names are different for each model)
        self._set_model_classifier(weights_cls, num_classes)

        self.preprocess = weights.transforms()
        self.loss = torch.nn.CrossEntropyLoss()

        

    def forward(self, x):
        return self.model(x)

    def on_validation_epoch_end(self):
        self.log('epoch', self.current_epoch)

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
        self.eval()
        img, label = batch
        x = self.preprocess(img)
        return self(x)

    '''
    def on_train_end(self):
        # save model on wandb
        self.logger.save()
        # save model on disk
        model_path = os.path.join(self.logger.log_dir, "model.pth")
        torch.save(self.state_dict(), model_path)
        self.logger.experiment.log_artifact(model_path)
    '''

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


    def _set_model_classifier(self, weights_cls, num_classes):
        weights_cls = str(weights_cls)
        if "ConvNeXt" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Flatten(1),
                torch.nn.Linear(self.model.classifier[2].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "EfficientNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[1].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "MobileNet" in weights_cls or "VGG" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[0].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "DenseNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier.in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "MaxVit" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.model.classifier[5].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "ResNet" in weights_cls or "RegNet" in weights_cls or "GoogLeNet" in weights_cls:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.fc.in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "Swin" in weights_cls:
            self.model.head = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.head.in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "ViT" in weights_cls:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.hidden_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "SqueezeNet1_1" in weights_cls or "SqueezeNet1_0" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
            )


    
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

