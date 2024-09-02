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

## Experiments

The experiments conducted in this project followed these key steps:

### 1. Training Pre-trained Networks

We utilized two popular convolutional neural network architectures, **ResNet18** and **VGG11**, to perform image classification tasks. The training was conducted under two different settings:

- **Feature Extraction**: In this setting, the pre-trained models were used to extract features from the images. The extracted features were then fed into a new classifier layer, which was trained from scratch. The rest of the network parameters were kept frozen.

- **Fine-tuning**: In this setting, we not only trained the new classifier layer but also fine-tuned all or some of the layers of the pre-trained models. This allowed the model to adjust its parameters more significantly based on the new dataset.

### 2. Generating and Evaluating Saliency Maps

After training the models, we focused on generating saliency maps to interpret the model's decision-making process. Saliency maps highlight the areas in an image that are most influential in the model's classification decisions. We evaluated the quality of these saliency maps using various metrics.

- **Saliency Map Generation**: We utilized four different post-hoc XAI (eXplainable AI) methods to generate saliency maps:

  1. **GradCAM (Gradient-weighted Class Activation Mapping)**: Uses gradients to identify areas that most influence the model's output.

  2. **RISE (Randomized Input Sampling for Explanation)**: Applies the concept of occlusion, masking parts of the image to observe the effect on prediction. 

  3. **SIDU (Saliency Interpretation with Decomposition and Unification)**: Combines occlusion and perturbation for more robust and detailed explanations.

  4. **LIME (Local Interpretable Model-agnostic Explanations)**: Utilizes perturbation, altering part of the image to assess the importance of each section.

- **Metrics Calculation**: The quality of the generated saliency maps was evaluated using the following metrics:

  1. **Deletion**: This metric evaluates how the classification confidence score changes when progressively removing the most salient regions indicated by the saliency map from the image. A high-quality saliency map should cause a significant drop in the confidence score as important regions are removed, reflecting the model’s reliance on these regions for its predictions.

  2. **Insertion**: This metric assesses how the classification confidence score changes when progressively adding the most salient regions back into a blank image. An effective saliency map should lead to a rapid recovery of the original confidence score as the important regions are reintroduced, demonstrating the saliency map’s ability to accurately highlight crucial image features.

These metrics provide a quantitative measure of how well the saliency maps identify the critical regions that influence the model's decisions.







