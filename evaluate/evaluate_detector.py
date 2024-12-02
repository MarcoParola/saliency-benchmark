import os

import hydra
from datasets import load_dataset

from src.datasets.detection import load_detection_dataset
from src.metrics.detector_metrics import DetectorMetrics
from src.models.detector.grounded_sam2 import GroundedSam2
from hydra.core.global_hydra import GlobalHydra

from src.models.detector.grounding_dino import GroundingDino
from datetime import datetime

if GlobalHydra().is_initialized():
    GlobalHydra().clear()


@hydra.main(config_path='../config', config_name='config')
def main(cfg):

    current_path = os.path.abspath(__file__)
    print(f"Path corrente: {current_path}")

    file_name="../../../evaluate/test_evaluate_detector.txt"

    with open(file_name, "a") as file:
        print("printing on file")
        file.write("------------------------------------------------TEST----------------------------------------------------------------------------------\n")
        file.write("Dataset:"+str(cfg.datasetDet)+"\n")
        file.write("Modello:" + str(cfg.modelDet) + "\n")
        file.write("Modello Sam:" + str(cfg.modelSam) + "\n")

    # Mostra il path assoluto del file
    absolute_path = os.path.abspath(file_name)
    print(f"Il file è stato salvato in: {absolute_path}")

    start_timestamp = datetime.now()
    # Retrieve dataset
    dataset = load_detection_dataset(cfg.datasetDet.name)

    # To evaluate GroundedSam2 performance I have to pass to him the classes of the dataset with which I perform the
    # comparison
    classes = '/'.join(dataset.classes)
    caption = classes
    if cfg.modelDet == 'GroundedSam2':
        if cfg.modelSam == 'Florence2':
            model = GroundedSam2(caption, 'Florence 2')  # call the model passing to it the caption
        elif cfg.modelSam == 'GroundingDino':
            model = GroundedSam2(caption, 'Grounding DINO')
    elif cfg.modelDet == 'GroundingDino':
        model = GroundingDino(caption)

    #Evaluate IoU and MAP
    detector_metrics = DetectorMetrics(model,dataset)
    avg_iou, avg_map, confidence_iou, confidence_map = detector_metrics()
    with open(file_name, "a") as file:
        file.write("Average IoU over the dataset:\n")
        file.write(f"{avg_iou}\n")
        file.write("Average MAP over the dataset:\n")
        file.write(f"{avg_map}\n")
        file.write("Confidence IoU over the dataset:\n")
        file.write(f"{confidence_iou}\n")
        file.write("Confidence MAP over the dataset:\n")
        file.write(f"{confidence_map}\n")

        endtime = datetime.now()
        duration = endtime - start_timestamp
        file.write("Duration: " + str(duration) + "\n")


if __name__ == "__main__":
    main()
