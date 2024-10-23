import hydra
from datasets import load_dataset

from src.datasets.detection import DetectionDataset
from src.metrics.detector_metrics import IntersectionOverUnion
from src.models.detector.grounded_sam2 import GroundedSam2

# @hydra.main(config_path='../config', config_name='config')
# def main(cfg):
#     # Retrieve dataset
#     dataset = DetectionDataset(load_dataset("Francesco/animals-ij5d2", split="test"))
#
#     # To evaluate GroundedSam2 performance I have to pass to him the classes of the dataset with which I perform the comparison
#     model = GroundedSam2(' '.join(dataset.classes))  # call the model passing to it the caption
#
#     # Evaluate IoU
#     average_iou = evaluate_iou(dataset, model)
#     print(f'Average IoU over the dataset: {average_iou:.4f}')

if __name__ == "__main__":
    #main()
    # Retrieve dataset
    dataset = DetectionDataset(load_dataset("Francesco/animals-ij5d2", split="test"))

    # To evaluate GroundedSam2 performance I have to pass to him the classes of the dataset with which I perform the comparison
    model = GroundedSam2(' '.join(dataset.classes))  # call the model passing to it the caption

    # Evaluate IoU
    detector_metrics = IntersectionOverUnion(model,dataset)
    average_iou = detector_metrics()
    print(f'Average IoU over the dataset: {average_iou:.4f}')