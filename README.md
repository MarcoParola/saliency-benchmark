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

## Usage

```sh
python train.py # TODO da vedere gli argomenti
```

```sh
python evaluate_method_acc.py model=resnet18 dataset.name=cifar10 method=gradcam
```
