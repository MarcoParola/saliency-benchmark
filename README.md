# saliency-benchmark
Repository for benchmarking different post-hoc xai explanation methods on image datasets

The main actions you can do are:
- train a DL model using `train.py` script
- assessing the XAI performance of a given model (trained by you or pre trained) on a specific dataset using a method for generating saliency maps by running `evaluate_method_acc.py` script  

## Install

To install the project, simply clone the repository and get the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/saliency-benchmark.git
cd saliency-benchmark
```

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```bash
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt
```

Next, create a new project on [Weights & Biases](https://wandb.ai/site) named `saliency-benchmark`. Edit `entity` parameter in [config.yaml](https://github.com/MarcoParola/saliency-benchmark/blob/main/config/config.yaml) by setting your wandb nick. Log in and paste your API key when prompted.
```sh
wandb login 
```

## Usage
A pretrained model fine-tuning can be run using `train.py` and specifying:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
 

```sh
python train.py model=ResNet18_Weights.IMAGENET1K_V1 dataset.name=oxford-iiit-pet
```

After fine-tuned a pre-trained model, you can reload it and evaluate its explainability by using `evaluate_method_acc.py`. Specify the following params:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`
- the `saliency.method` param from the following: `sidu`, `gradcam`, `lime`, `rise`.
- the `checkpoint` param by choosing among the pretrained model checkpoints in the output folder. Pleas note, in the following example the `checkpoint` param is valued according the windows path format.

Please note, `evaluate_method_acc.py` requires a target layer depending on the model and the saliency method. They are declared in `config\target_layers.yaml`. Edit this configuration file to set different target layers.

```sh
python evaluate_method_acc.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=cifar10 saliency.method=gradcam checkpoint=outputs\VGG11\epoch\=0-step\=2500.ckpt
```



python evaluate_method_acc.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=cifar10 saliency.method=sidu checkpoint=outputs/VGG11/VGG11.ckpt




|  **CIFAR10**    | GradCam |       | SIDU  |       |
|-----------------|---------|-------|-------|-------|
|                 | Ins     | Del   | Ins   | Del   |
| VGG1            | 12.16   | 10.12 | 12.21 | 10.13 |
| EfficientNet_B0 |         |       |       |       |

