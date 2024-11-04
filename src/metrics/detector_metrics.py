import numpy as np
import torch
from torch import tensor
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

from src.utils import from_array_to_dict, from_array_to_dict_predicted, save_annotated_images, save_annotated


class DetectorMetrics:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def __call__(self):
        """
           Evaluates the IoU for the given dataset using the model predictions.

           Parameters:
               dataset: the dataset containing the test samples
               model: the model used for generating predictions

           Returns:
               average_iou: float, average IoU over the entire dataset
        """

        for idx in range(self.dataset.__len__()):
            # Get the ground truth bounding boxes and labels
            image, ground_truth_boxes, ground_truth_labels = self.dataset.__getitem__(idx)

            # Predict using the model
            box_predicted, label_predicted, scores_predicted = self.model(image)

            #Mapping of the predicted label and the true ones
            list_ontology = self.model.ontology.classes()
            ground_truth_labels_elements = [self.dataset.classes[i] for i in ground_truth_labels]
            ground_truth_labels = [list_ontology.index(elem) for elem in ground_truth_labels_elements]

            #Update metric

            self.metric.update(from_array_to_dict_predicted(box_predicted, scores_predicted, label_predicted),
                               from_array_to_dict(ground_truth_boxes, ground_truth_labels))

            if idx<=100:
                label_predicted_elements=[list_ontology[i] for i in label_predicted]
                save_annotated(image, ground_truth_boxes, ground_truth_labels_elements,box_predicted,
                               label_predicted_elements,scores_predicted,label_predicted, idx)


        # Calculate average metric
        average_iou = self.metric.compute()

        self.metric.reset()

        return average_iou


class IoU(DetectorMetrics):
    def __init__(self, model, dataset):
        super().__init__(model,dataset)
        self.metric = IntersectionOverUnion(box_format='xyxy', iou_threshold=0.5, respect_labels=False)


class MAP(DetectorMetrics):
    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self.metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')


# Example usage
if __name__ == "__main__":
    # # Create the IoU instance
    # iou_metric = IntersectionOverUnion(class_metrics=True, iou_threshold=0.5)
    #
    # preds = [
    #     {
    #         "boxes": torch.tensor([
    #             [296.55, 93.96, 314.97, 152.79],
    #             [298.55, 98.96, 314.97, 151.79]]),
    #         "labels": torch.tensor([4, 5]),
    #     }
    # ]
    # target = [
    #     {"boxes": torch.tensor([
    #         [300.00, 100.00, 315.00, 150.00],
    #         [300.00, 100.00, 315.00, 150.00]
    #     ]),
    #         "labels": torch.tensor([4, 5]),
    #     }
    # ]
    #
    # # Update the metric with predictions and ground truth
    # iou_value = iou_metric(preds, target)
    #
    # print(f"IoU: {iou_value}")
    #
    # iou_metric.update(preds, target)
    #
    # preds = [
    #     {
    #         "boxes": torch.tensor([
    #             [150.25, 75.50, 200.75, 125.30],
    #             [210.00, 100.00, 250.00, 180.00]
    #         ]),
    #         "labels": torch.tensor([2, 3]),
    #     }
    # ]
    # target = [
    #     {
    #         "boxes": torch.tensor([
    #             [160.00, 80.00, 205.00, 130.00],
    #             [215.00, 110.00, 255.00, 185.00]
    #         ]),
    #         "labels": torch.tensor([2, 3]),
    #     }
    # ]
    #
    # # Update the metric with predictions and ground truth
    # iou_value = iou_metric(preds, target)
    #
    # print(f"IoU: {iou_value}")
    #
    # iou_metric.update(preds, target)
    #
    # iou_value = iou_metric.compute()
    # print(f"IoU: {iou_value}")

    # prova Mean Average Precision

    preds = [dict(boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
                  scores=tensor([0.536]),
                  labels=tensor([0]),
                  )
             ]
    target = [dict(boxes=tensor([[214.0, 41.0, 562.0, 285.0]]), labels=tensor([0]))]

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.update(preds, target)
    metric_value = metric.compute()
    print(f"IoU: {metric_value}")
