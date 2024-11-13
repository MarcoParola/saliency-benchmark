import hydra
from datasets import load_dataset

from src.datasets.detection import load_detection_dataset
from src.metrics.detector_metrics import DetectorMetrics
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
        if cfg.modelSam == 'Florence2':
            model = GroundedSam2(caption,'Florence 2')  # call the model passing to it the caption
        elif cfg.modelSam == 'GroundingDino':
            model = GroundedSam2(caption,'Grounding DINO')
    elif cfg.model == 'GroundingDino':
        model = GroundingDino(caption)

    #Evaluate IoU and MAP
    detector_metrics = DetectorMetrics(model,dataset)
    avg_iou, avg_map = detector_metrics()
    print("Average IoU over the dataset: ")
    print(avg_iou)
    print("Average MAP over the dataset: ")
    print(avg_map)




if __name__ == "__main__":
    main()
