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

import hydra
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def plot_auc_curve(x, y, method_name):
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
        self.model = model
        self.n_pixels = n_pixels

    def __call__(self, image, saliency_map, class_label, start_with_blurred):
        h, w = saliency_map.shape

        if start_with_blurred:
            working_image = torchvision.transforms.functional.gaussian_blur(image, kernel_size=[127, 127],
                                                                            sigma=[17, 17]).clone()
            method_name = "insertion"
        else:
            working_image = image.clone()
            method_name = "deletion"

        # Flatten saliency map and sort the indices in descending order
        sorted_index = torch.flip(saliency_map.view(-1).argsort(), dims=[0])

        start_idx = 0
        iteration = 0
        predictions = []

        prediction = self.model(image.unsqueeze(0))
        class_score = prediction[0, class_label].item()

        predictions.append(class_score)

        while start_idx < h * w:
            # Select the next n_pixels indices starting from start_idx
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

            # Make a prediction on the masked image and store the score for the predicted class
            prediction = self.model(working_image.unsqueeze(0))
            class_score = prediction[0, class_label].item()
            predictions.append(class_score)

            iteration += 1

        # Calculate the AUC
        x = [i / (len(predictions) - 1) for i in range(len(predictions))]
        auc_score = auc(x, predictions)

        # Plot AUC curve
        plot_auc_curve(x, predictions, method_name)

        return auc_score

    def generate_masked_image(self, *args, **kwargs):
        raise NotImplementedError


class Insertion(SaliencyMetrics):
    def __init__(self, model, n_pixels):
        super(Insertion, self).__init__(model, n_pixels)

    def generate_masked_image(self, index, image, working_image, h, w):
        mask = torch.zeros(h * w, device=image.device)
        mask[index] = 1
        mask = mask.view(h, w)

        # Sostituisci i pixel dell'immagine sfocata con quelli dell'immagine originale
        masked_image = torch.where(mask == 1, image, working_image)

        return masked_image


class Deletion(SaliencyMetrics):
    def __init__(self, model, n_pixels):
        super(Deletion, self).__init__(model, n_pixels)

    def generate_masked_image(self, index, image, working_image, h, w):
        # Create a mask initialized to 1
        mask = torch.ones(h * w, device=image.device)

        # Set the selected saliency pixels in the mask to 0
        mask[index] = 0

        # Reshape the mask to the original image shape
        mask = mask.view(h, w)

        # Apply the mask to the image
        masked_image = working_image * mask

        # Optional: Return the modified image
        return masked_image
