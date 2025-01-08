import os

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from torchvision.transforms import ToPILImage

from src.datasets.classification import load_classification_dataset
from src.models.detector.grounded_sam2 import GroundedSam2
from src.utils import save_mask, load_mask

if GlobalHydra().is_initialized():
    GlobalHydra().clear()


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    caption = ("Very defined erythematous edges/Uneven grey ulcer ground/Granulating and necrotic ulcer background/ "
               "White Fibrin/ Marked erythematous edges")
    # IMAGE_PATH = "images/flower.jpg"

    resize = 224

    train, val, test = load_classification_dataset(cfg.dataset.name, "data", resize)

    dataset = val

    torch.cuda.empty_cache()

    if cfg.modelSam == 'Florence2':
        model = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
    elif cfg.modelSam == 'GroundingDino':
        model = GroundedSam2(caption, 'Grounding DINO')

    for idx in range(2):
        print("IMG " + str(idx))
        # Get the ground truth bounding boxes and labels
        image, ground_truth_labels = dataset.__getitem__(idx)

        image = ToPILImage()(image)

        # Predict using the model
        boxes, masks, classes = model.mask_generation_with_all_concepts(image, resize)

        # print(masks.shape)

        if len(masks) > 0:
            # save_images_with_mask(boxes,masks,classes,image,model,idx) #usable in case of printing only present class mask
            # save_images_with_mask_for_all_concepts(image, masks, model, boxes)  #to print all the class masks, even if not present
            # plot_grid_masks(image, masks, model.ontology.classes(), idx)
            output_dir_tensors = os.path.join(os.path.abspath('mask_output'), cfg.dataset.name)
            if not os.path.exists(output_dir_tensors):
                os.makedirs(output_dir_tensors)
            print(masks)
            save_mask(os.path.join(output_dir_tensors, f'mask_{idx}.pth'), masks)


if __name__ == '__main__':
    main()
    # print("Riprova")
    # directory_path = 'C:/Users/matte/GitHub/saliency-benchmark/mask_output/oral'
    # for filename in os.listdir(directory_path):
    #     file_path = os.path.join(directory_path, filename)
    #     if os.path.isfile(file_path):
    #         print(file_path)
    #         saliency_map = load_mask(file_path)
    #         print(saliency_map.dtype)
    #         print(saliency_map)
