# saliency-benchmark
Repository for benchmarking different post-hoc XAI explanation methods on image datasets. Here is a quick guide on how to install and use the repo. More information about installation and usage can be found in the [documentation](docs/README.md).

## Install
To install the project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/saliency-benchmark.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd saliency-benchmark
    ```

3. **Create a virtual environment:**

    ```bash
    python -m venv env
    ```

4. **Activate the virtual environment:**

    ```bash
    . env/bin/activate
    ```

5. **Install the dependencies:**

    ```bash
    python -m pip install -r requirements.txt
    ```

These steps will set up your working environment, install necessary dependencies, and prepare you to run the project.

## Training

To train the networks using this repository, use the following command:

```bash
python3 train.py model=VGG11_Weights.IMAGENET1K_V1 dataset.name=cifar10 train.finetune=True
```
- `model`: Specifies the pre-trained model to use. The full list of available models can be found [here](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification

- `dataset.name`: Specifies the dataset to use. The supported datasets are:
    - `cifar10`
    - `cifar100`
    - `caltech101`
    - `mnist`
    - `svhn`
    - `oxford-iiit-pet`
  
- `train.finetune`: Determines the training mode.
    - `True`: Fine-tunes the entire model.
    - `False`: Uses the model as a feature extractor.

These parameters allow you to customize the training process according to your specific requirements. For a detailed configuration, you may refer to or modify the train section of the [config.yaml](config/config.yaml) file according to your specific requirements.

## Testing 

To evaluate the trained model, use the following command:

```bash
python3 test.py
```
You need to specify the following parameters in the config.yaml file:
- `model`: The pre-trained model to use.
- `dataset.name`: The dataset used for testing.
- `checkpoint`: Path to the model checkpoint. Choose from the model checkpoints available in the **checkpoints** folder.
