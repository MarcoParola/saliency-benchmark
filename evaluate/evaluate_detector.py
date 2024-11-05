import hydra
from datasets import load_dataset

from src.datasets.detection import load_detection_dataset
from src.metrics.detector_metrics import IoU, MAP
from src.models.detector.grounded_sam2 import GroundedSam2
from hydra.core.global_hydra import GlobalHydra

from src.models.detector.grounding_dino import GroundingDino

if GlobalHydra().is_initialized():
    GlobalHydra().clear()


@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    # Retrieve dataset
    dataset = load_detection_dataset(cfg.dataset.name)

    # To evaluate GroundedSam2 performance I have to pass to him the classes of the dataset with which I perform the comparison
    classes = '/'.join(dataset.classes)
    caption = classes
    if cfg.model == 'GroundedSam2':
        model = GroundedSam2(caption)  # call the model passing to it the caption
    elif cfg.model == 'GroundingDino':
        model = GroundingDino(caption)

    if cfg.metrics.name == 'iou':
        # Evaluate IoU
        detector_metrics = IoU(model, dataset)
        #print(dataset.classes)
        average_iou = detector_metrics()
        print(f'Average IoU over the dataset: ')
        print(average_iou)

    elif cfg.metrics.name == 'map':
        #Evaluate MAP
        print(model.caption)
        detector_metrics = MAP(model, dataset)
        print(dataset.classes)
        avg_map = detector_metrics()
        print(f'Average MAP over the dataset: ')
        print(avg_map)


if __name__ == "__main__":
    main()
