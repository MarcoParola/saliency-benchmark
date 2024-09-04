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
### configuration options 

