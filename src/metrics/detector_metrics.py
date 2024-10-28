# def calculate_iou(box1, box2):
#     """
#     Calculate the Intersection over Union (IoU) of two bounding boxes in COCO format.
#     Boxes are defined as [x, y, w, h].
#
#     Parameters:
#         box1: list or array, bounding box 1 [x, y, w, h]
#         box2: list or array, bounding box 2 [x, y, w, h]
#
#     Returns:
#         iou: float, Intersection over Union value
#     """
#     # Extract coordinates and dimensions
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2
#
#     # Calculate the intersection coordinates
#     x_left = max(x1, x2)
#     y_top = max(y1, y2)
#     x_right = min(x1 + w1, x2 + w2)
#     y_bottom = min(y1 + h1, y2 + h2)
#
#     # Check if there is an intersection
#     if x_right < x_left or y_bottom < y_top:
#         return 0.0
#
#     # Area of intersection
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)
#
#     # Area of both bounding boxes
#     box1_area = w1 * h1
#     box2_area = w2 * h2
#
#     # Area of union
#     union_area = box1_area + box2_area - intersection_area
#
#     # Compute IoU
#     iou = intersection_area / union_area
#     return iou
#
#
# # The Bbox must be passed in the COCO format [x,y,w,h]
# def convert_to_coco_format(pred_box):
#     # the input is in format x_top_left,y_top_left,x_bottom_right,y_bottom_right and I want in format x,y,w,h
#     new_box = pred_box.copy()
#     new_box[2] = abs(pred_box[2] - pred_box[0])
#     new_box[3] = abs(pred_box[1] - pred_box[3])
#     return new_box
#
#
# class IntersectionOverUnion:
#     def __init__(self, model, dataset):
#         self.model = model
#         self.dataset = dataset
#
#     def __call__(self):
#         """
#            Evaluates the IoU for the given dataset using the model predictions.
#
#            Parameters:
#                dataset: the dataset containing the test samples
#                model: the model used for generating predictions
#
#            Returns:
#                average_iou: float, average IoU over the entire dataset
#            """
#         iou_scores = []
#
#         for idx in range(self.dataset.__len__()):
#             # Get the ground truth bounding boxes and labels
#             image, ground_truth_boxes, ground_truth_labels = self.dataset.__getitem__(idx)
#
#             # Predict using the model
#             box_predicted, label_predicted = self.model(image)
#
#             # Compare each predicted bounding box to the ground truth boxes
#             for pred_box, pred_label in zip(box_predicted, label_predicted):
#                 # Find the matching ground truth box (if any)
#                 matching_boxes = [
#                     calculate_iou(convert_to_coco_format(pred_box), gt_box)
#                     for gt_box, gt_label in zip(ground_truth_boxes, ground_truth_labels)
#                     if pred_label == gt_label
#                 ]
#
#                 # If there are matches, take the highest IoU
#                 if matching_boxes:
#                     best_iou = max(matching_boxes)
#                     iou_scores.append(best_iou)
#
#         # Calculate average IoU
#         average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
#         return average_iou

import torch
from torch import tensor
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

from src.utils import from_array_to_tensor, from_array_to_dict


class IoU:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.metric = IntersectionOverUnion(box_format='xyxy', iou_threshold=0.5, respect_labels=False)

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

            self.metric.update(from_array_to_tensor(box_predicted, label_predicted),
                               from_array_to_tensor(ground_truth_boxes, ground_truth_labels))

        # Calculate average IoU
        average_iou = self.metric.compute()
        return average_iou


class MAP:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    def __call__(self):
        for idx in range(self.dataset.__len__()):
            # Get the ground truth bounding boxes and labels
            image, ground_truth_boxes, ground_truth_labels = self.dataset.__getitem__(idx)

            # Predict using the model
            box_predicted, label_predicted, scores_predicted = self.model(image)

            self.metric.update(from_array_to_dict(box_predicted, scores_predicted, label_predicted),
                               from_array_to_tensor(ground_truth_boxes, ground_truth_labels))

        # Calculate average IoU
        average_map = self.metric.compute()
        return average_map


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
