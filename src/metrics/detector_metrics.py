import numpy as np
import torch
from torch import tensor
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchvision.ops import box_convert

from src.utils import from_array_to_dict, from_array_to_dict_predicted, save_annotated_images, save_annotated, \
    save_annotated_grounding_dino, from_normalized_cxcywh_to_xyxy


class DetectorMetrics:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        # class_metric=True should add also the metric for each class
        self.iou = IntersectionOverUnion(box_format='xyxy', iou_threshold=0.5, respect_labels=False, class_metrics=True)
        self.map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

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
            print("IMG "+str(idx))
            # Get the ground truth bounding boxes and labels
            image, ground_truth_boxes, ground_truth_labels = self.dataset.__getitem__(idx)
            print("Ground truth labels: ", ground_truth_labels)

            # Predict using the model
            box_predicted, label_predicted, scores_predicted = self.model(image)
            print("label_predicted: ", label_predicted)
            #print("Predicted boxes: ", box_predicted)

            #Mapping of the predicted label and the true ones
            #GroundedSam2
            list_ontology = self.model.ontology.classes()
            ground_truth_labels_elements = [self.dataset.classes[i] for i in ground_truth_labels]
            ground_truth_labels = [list_ontology.index(elem) for elem in ground_truth_labels_elements]
            label_predicted_elements = [list_ontology[i] for i in label_predicted]

            #Update metric

            if len(ground_truth_boxes)>0:
                self.iou.update(from_array_to_dict_predicted(box_predicted, scores_predicted, label_predicted),
                               from_array_to_dict(ground_truth_boxes, ground_truth_labels))

                self.map.update(from_array_to_dict_predicted(box_predicted, scores_predicted, label_predicted),
                               from_array_to_dict(ground_truth_boxes, ground_truth_labels))

            # if idx<=100:
            #     #GROUNDEDSAM2 model
            #     save_annotated(image, ground_truth_boxes, ground_truth_labels_elements,box_predicted,
            #                     label_predicted_elements,scores_predicted,label_predicted,ground_truth_labels, idx)
            # else:
            #     break
            if idx>100:
                 break

        # Calculate average metric
        average_iou = self.iou.compute()
        average_map = self.map.compute()

        self.iou.reset()
        self.map.reset()

        return average_iou, average_map


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
