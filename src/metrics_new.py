import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn


#HW = 224 * 224  # image area
#n_classes = 1000

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric:

    def __init__(self, model, mode, step, substrate_fn, dim, n_classes):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.dim = dim
        self.n_classes = n_classes

    '''
    def single_run(self, img_tensor, explanation, verbose=2, save_to=None):
        # Print the initial shape of the input tensor
        print(f"Initial img_tensor shape: {img_tensor.shape}")

        # Ensure the input tensor is on the right device
        # img_tensor = img_tensor.cuda()
        pred = self.model(img_tensor)

        # Print the shape after passing through the model
        print(f"Shape after model: {pred.shape}")

        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (self.img_size + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        salient_order = np.flip(np.argsort(explanation.reshape(-1, self.img_size), axis=1), axis=-1)
        for i in range(n_steps + 1):
            pred = self.model(start.cuda())
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, self.img_size)[0, :, coords] = finish.cpu().numpy().reshape(1, 3,
                                                                                                              self.img_size)[
                                                                                 0, :, coords]
        return scores
    '''

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, self.n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i * batch_size:(i + 1) * batch_size]).cpu()
            predictions[i * batch_size:(i + 1) * batch_size] = preds
        top = np.argmax(predictions.detach().numpy(), -1) # per ogni immagine del batch viene presa la classe con la probabilità più alta
        n_steps = (self.dim + self.step - 1) // self.step #quanti step servono per alterare tutti i pixel dell'immagine
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.cpu().detach().numpy().reshape(-1, self.dim), axis=1), axis=-1) #matrice che contiene gli indici dei pixel ordinati in ordine decrescente di importanza per ciascuna immagine nel batch
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch) #maschera
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j * batch_size:(j + 1) * batch_size] = self.substrate_fn(
                img_batch[j * batch_size:(j + 1) * batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps + 1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j * batch_size:(j + 1) * batch_size])
                preds = preds.detach().cpu().numpy()[range(batch_size), top[j * batch_size:(j + 1) * batch_size]]
                scores[i, j * batch_size:(j + 1) * batch_size] = preds
                print(f" Class Index {top[j * batch_size:(j + 1) * batch_size]}")
                print(f"Prediction score {preds}")
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, self.dim)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3,
                                                                                                       self.dim)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))

        return auc(scores.mean(1)), scores