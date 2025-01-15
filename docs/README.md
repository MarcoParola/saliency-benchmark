# saliency-benchmark
Documentation for installing and using the repo.

## Install

```sh
# Clone repository
git clone https://github.com/MarcoParola/saliency-benchmark.git
cd saliency-benchmark

# Create virtual environment and install dependencies
python -m venv env
env/Scripts/activate
./setup.bat

# Weight&Biases login
wandb login 
```

## Usage

### Training
A pretrained model fine-tuning can be run using `train.py` and specifying in the [config.yaml](config/config.yaml) file:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`, `imagenette`, `intel_image`

```sh
python train.py model=ResNet18_Weights.IMAGENET1K_V1 dataset.name=oxford-iiit-pet
```

### Testing
After fine-tuned a pre-trained model, you can reload it and evaluate it by using `test.py`. Specify the following params in the [config.yaml](config/config.yaml) file:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`, `imagenette`, `intel_image`
- the `checkpoint` param by choosing among the pretrained model checkpoints in the output folder. Pleas note, in the following example the `checkpoint` param is valued according the windows path format.

```sh
python test.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=cifar10 checkpoint=checkpoints\finetuned_VGG11_Weights.IMAGENET1K_V1intel_image\model-epoch\=13-val_loss\=1.12.ckpt
```
In the test step, it is possible to obtain a confusion matrix, running the [confusion_matrix_generation.py](scripts/confusion_matrix_generation.py) script.

### Generate saliency map
After training and testing the model, you can generate the saliency maps using `generate_saliency.py`. Specify the following params in the [config.yaml](config/config.yaml) file:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`, `imagenette`, `intel_image`
- the `checkpoint` param by choosing among the pretrained model checkpoints in the output folder. Pleas note, in the following example the `checkpoint` param is valued according the windows path format.
- the `saliency.method` param from the following: `sidu`, `gradcam`, `lime`, `rise`, `lrp`.
- the `saliency.dataset` boolean param in order to choose if produce the saliency map for all the image of the specified dataset (setting it to True), or only for a specified image (setting it to False). For this second possibility, the image has to be put in the [image](data/image) folder

### Extract concept
To extract the concept from a specific dataset:
- You can define the concepts for a specific dataset in a csv file `{dataset_name}_concepts.csv`, saved in the [concepts](data/concepts) folder, in which each row is of the type "class,concept1;concept2;...".
- Then you can extract the concept using `extract_concept.py`, specifying the following params:
  - the `modelSam` param from the following `GroundingDino`,`Florence2`
  - the `dataset.name` param from the following list: `imagenette`, `intel_image`
  - the `mask.dataset` boolean param in order to choose if extract the concepts for all the image of the specified dataset (setting it to True), or only for a specified image (setting it to False). For this second possibility, the image has to be put in the [image](data/image) folder

### WoE


