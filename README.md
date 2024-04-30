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
- the `dataset.name` param from ... TODO prepare a list of supported dataset? 

```sh
python train.py model=ResNet18_Weights.IMAGENET1K_V1 dataset.name=oxford-iiit-pet
```

```sh
python evaluate_method_acc.py model=resnet18 dataset.name=cifar10 method=gradcam
```
