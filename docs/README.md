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
mkdir data
mkdir data/image

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
- the `checkpoint` param by choosing among the pretrained model checkpoints in the checkpoint folder. Please note, in the following example the `checkpoint` param is valued according the windows path format.

```sh
python test.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=intel_image checkpoint=checkpoints\finetuned_VGG11_Weights.IMAGENET1K_V1intel_image\model-epoch\=13-val_loss\=1.12.ckpt
```
In the test step, it is possible to obtain a confusion matrix, running the [confusion_matrix_generation.py](scripts/confusion_matrix_generation.py) script, specifying the same parameters mentioned above.

### Generate saliency map
After training and testing the model, you can generate the saliency maps using `generate_saliency.py`. Specify the following params in the [config.yaml](config/config.yaml) file:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `cifar10`, `cifar100`, `caltech101`, `imagenet`, `oxford-iiit-pet`, `svhn`, `mnist`, `fashionmnist`, `imagenette`, `intel_image`
- the `checkpoint` param by choosing among the pretrained model checkpoints in the checkpoint folder. Please note, in the following example the `checkpoint` param is valued according the windows path format.
- the `saliency.method` param from the following: `sidu`, `gradcam`, `lime`, `rise`.
- the `saliency.dataset` boolean param in order to choose if produce the saliency map for all the image of the specified dataset (setting it to True), or only for a specified image (setting it to False). For this second possibility, the image has to be put in the [image](data/image) folder, specifying a parameter:
    - the `saliency.file_image` param, in which you have to specify the name of the file in which the image is saved

To generate the saliency maps for all the images of the specified dataset use the following command:
```sh
python generate_saliency.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=intel_image checkpoint=checkpoints\finetuned_VGG11_Weights.IMAGENET1K_V1intel_image\model-epoch\=13-val_loss\=1.12.ckpt saliency.method=gradcam saliency.dataset=True
```

To generate the saliency map for the specified image, use the following command:
```sh
python generate_saliency.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=intel_image checkpoint=checkpoints\finetuned_VGG11_Weights.IMAGENET1K_V1intel_image\model-epoch\=13-val_loss\=1.12.ckpt saliency.method=gradcam saliency.dataset=False saliency.file_image="bird.jpg"
```

### Extract concept
To extract the concept from a specific dataset:
- You can define the concepts for a specific dataset in a csv file `{dataset_name}_concepts.csv`, saved in the [concepts](data/concepts) folder, in which each row is of the type "class,concept1;concept2;...".
- Then you can extract the concept using `extract_concept.py`, specifying the following params:
  - the `modelSam` param from the following `GroundingDino`,`Florence2`
  - the `dataset.name` param from the following list: `imagenette`, `intel_image`
  - the `mask.dataset` boolean param in order to choose if extract the concepts for all the image of the specified dataset (setting it to True), or only for a specified image (setting it to False). For this second possibility, the image has to be put in the [image](data/image) folder, specifying two parameters:
    - the `mask.concepts` param, in which you can put all the concepts that you want to extract, divided by a "/", like visible in the following example
    - the `mask.file_image` param, in which you have to specify the name of the file in which the image is saved

To extract concepts from all the images of the specified dataset, use the following command:
```sh
python extract_concept.py modelSam=GroundingDino dataset.name=intel_image mask.dataset=True 
```
To extract concepts from the specified image, use the following command:
```sh
python extract_concept.py modelSam=GroundingDino mask.dataset=False mask.concepts="beak/feathers/eyes" mask.file_image="bird.jpg" 
```
### Weight of Evidence
You can compute weight of evidence score using file `evaluate/woe_evaluation.py`, to produce the weight of evidence score for a specific combination of model and methods you have to specify the following parameters:
- the `model` param from the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- the `dataset.name` param from the following list: `imagenette`, `intel_image`
- the `checkpoint` param by choosing among the pretrained model checkpoints in the checkpoint folder. Please note, in the following example the `checkpoint` param is valued according the windows path format.
- the `saliency.method` param from the following: `sidu`, `gradcam`, `lime`, `rise`.
- the `modelSam` param from the following `GroundingDino`,`Florence2`.
- the `woe.concept_presence_method` param from the following `cas`, `iou`, `casp`.
- the `woe.concept_favor_against` boolean param in order to choose if compute woe score considering all the concepts for each class, or if compute the woe score for the 'favor' and 'against' concepts for a specific class, that are needed to be define in a csv file , saved as `{dataset_name}_concepts_favor_against.csv`, saved in the [concepts](data/concepts) folder.

To produce woe score considering each classes of the dataset and each concepts defined, use the following command:
```sh
python -m evaluate.woe_evaluation model=ResNet18_Weights.IMAGENET1K_V1 dataset.name=imagenette modelSam=GroundingDino saliency.method=gradcam checkpoint=checkpoints\finetuned_ResNet_imagenette.ckpt woe.concept_presence_method=cas woe.concept_favor_against=False
```

### Fidelity metrics for saliency map

You can evaluate the explainability of the model by using fidelity metrics using the following command: 
```bash
python -m evaluate.evaluate_saliency
```
You need to specify the following parameters in the [config.yaml](config/config.yaml) file:
- `model`: The pre-trained model to use.
- `dataset.name`: The dataset used for testing.
- `checkpoint`: Path to the model checkpoint. Choose from the model checkpoints available in the **checkpoints** folder.
- `saliency.method`: Saliency method used for evaluating the model's explanations. The supported methods are: `gradcam`, `rise`, `sidu`, `lime`.
