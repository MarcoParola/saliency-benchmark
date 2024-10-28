import hydra
from datasets import load_dataset

from src.datasets.detection import load_detection_dataset
from src.metrics.detector_metrics import IoU, MAP
from src.models.detector.grounded_sam2 import GroundedSam2
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra().is_initialized():
    GlobalHydra().clear()


@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    # Retrieve dataset
    dataset = load_detection_dataset(cfg.dataset.name)

    # To evaluate GroundedSam2 performance I have to pass to him the classes of the dataset with which I perform the comparison
    classes = ' '.join(dataset.classes)
    caption = "Retrieve me the following classes: "+ classes
    model = GroundedSam2(caption)  # call the model passing to it the caption

    # Evaluate IoU
    # detector_metrics = IoU(model, dataset)
    # average_iou = detector_metrics()
    # print(f'Average IoU over the dataset: ')
    # print(average_iou)

    #Evaluate MAP
    detector_metrics = MAP(model, dataset)
    avg_map = detector_metrics()
    print(f'Average MAP over the dataset: ')
    print(avg_map)


if __name__ == "__main__":
    main()
