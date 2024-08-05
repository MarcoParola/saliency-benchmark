'''
    - input:
        -modello: Rete Neurale preaddestrata
        -immagini: set di immagini da usare per la valutazione
        -metodo di generazione: metodo usato pe generare le mappe di salienza

    -step:
        0. Inizializzazione delle metriche a zero
        1. dato modello, metodo e immagine generare la mappa di salienza
        2. loop dove si calcolare le maschere con le tecniche di oscuramento
        3. predizione dell'immagini mascherate
        4. aggiornamento dei valori parziali delle metriche
'''

import os
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def save_auc_curve(x, y, method_name):
    """Plot and save the AUC curve for the saliency method."""
    plt.figure()
    plt.plot(x, y, label=f'AUC = {auc(x, y):.2f}')
    plt.xlabel('Fraction of top pixels')
    plt.ylabel('Model score')
    plt.title(f'{method_name.capitalize()} AUC Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(f'{method_name}_auc_curve.png'))
    plt.close()


class SaliencyMetrics:
    def __init__(self, model, n_pixels):
        """Initialize with the model and number of pixels to modify per iteration."""
        self.model = model
        self.n_pixels = n_pixels

    def __call__(self, image, saliency_map, class_label, start_with_blurred):
        """Evaluate the saliency map using insertion or deletion method."""
        h, w = saliency_map.shape

        # Choose the starting image: blurred or original
        if start_with_blurred:
            working_image = torchvision.transforms.functional.gaussian_blur(
                image, kernel_size=[127, 127], sigma=[17, 17]
            ).clone()
            method_name = "insertion"
        else:
            working_image = image.clone()
            method_name = "deletion"

        # Sort the saliency map indices in descending order
        sorted_index = torch.flip(saliency_map.view(-1).argsort(), dims=[0])

        start_idx = 0
        iteration = 0
        predictions = []

        # Initial prediction score for the original image
        prediction = self.model(image.unsqueeze(0))
        class_score = prediction[0, class_label].item()
        predictions.append(class_score)

        # Iteratively mask the image and record model predictions
        while start_idx < h * w:
            end_idx = min(start_idx + self.n_pixels, h * w)
            top_indices = sorted_index[start_idx:end_idx]

            working_image = self.generate_masked_image(top_indices, image, working_image, h, w)
            start_idx += self.n_pixels  # Move to the next set of pixels to mask

            '''
            # Plot and save the masked image
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            masked_image_np = working_image.permute(1, 2, 0).cpu().numpy()
            ax.imshow(masked_image_np)
            ax.set_title(f'Masked Image Iteration {iteration}')
            ax.axis('off')

            # Save the masked image
            plt.savefig(os.path.join(f'{method_name}_image_iter_{iteration}.png'))
            plt.close(fig)
            '''

            # Make a prediction on the masked image
            prediction = self.model(working_image.unsqueeze(0))
            class_score = prediction[0, class_label].item()
            predictions.append(class_score)

            iteration += 1

        # Calculate the AUC
        x = [i / (len(predictions) - 1) for i in range(len(predictions))]
        auc_score = auc(x, predictions)

        # Save the AUC curve
        save_auc_curve(x, predictions, method_name)

        return auc_score

    def generate_masked_image(self, *args, **kwargs):
        """Generate the masked image based on the selected method (to be implemented in subclasses)."""
        raise NotImplementedError


class Insertion(SaliencyMetrics):
    def __init__(self, model, n_pixels):
        super(Insertion, self).__init__(model, n_pixels)

    def generate_masked_image(self, index, image, working_image, h, w):
        """Mask the most salient pixels by copying them from the original image to the blurred image."""
        mask = torch.zeros(h * w, device=image.device)
        mask[index] = 1
        mask = mask.view(h, w)

        # Replace the pixels in the blurred image with the original image's pixels
        masked_image = torch.where(mask == 1, image, working_image)

        return masked_image


class Deletion(SaliencyMetrics):
    def __init__(self, model, n_pixels):
        super(Deletion, self).__init__(model, n_pixels)

    def generate_masked_image(self, index, image, working_image, h, w):
        """Mask the most salient pixels by setting them to zero in the working image."""
        mask = torch.ones(h * w, device=image.device)
        mask[index] = 0
        mask = mask.view(h, w)

        # Apply the mask to the working image
        masked_image = working_image * mask

        return masked_image
