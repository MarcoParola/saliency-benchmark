import numpy as np
import torch
from torch import tensor
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage

from src.utils import from_array_to_dict, from_array_to_dict_predicted

import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h

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

            # Predict using the model
            box_predicted, label_predicted, scores_predicted = self.model(ToPILImage()(image))

            #Mapping of the predicted label and the true ones
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
            # if idx>1000:
            #      break

        # Calculate average metric
        average_iou = self.iou.compute()
        iou_class_list = [float(value) for key, value in average_iou.items() if 'iou/cl_' in key]
        mean_iou,min_iou,max_iou, confidence_interval_iou = mean_confidence_interval(iou_class_list)
        average_map = self.map.compute()
        map_class_list = np.array(average_map['map_per_class'])
        mean_map, min_map, max_map, confidence_interval_map = mean_confidence_interval(map_class_list)


        with open("../../../evaluate/test_evaluate_detector.txt", "a") as file:
            file.write("iou class list:"+str(iou_class_list)+"\n")
            file.write("mean_iou:"+str(mean_iou)+"\n")
            file.write("min_iou:" + str(min_iou)+"\n")
            file.write("max_iou:" + str(max_iou)+"\n")
            file.write("confidence iou:"+ str(confidence_interval_iou)+"\n")
            file.write("map class list:" + str(map_class_list)+"\n")
            file.write("mean_map:" + str(mean_map)+"\n")
            file.write("min_map:" + str(min_map)+"\n")
            file.write("max_map:" + str(max_map)+"\n")
            file.write("confidence map:" + str(confidence_interval_map)+"\n")



        self.iou.reset()
        self.map.reset()

        return average_iou, average_map, confidence_interval_iou, confidence_interval_map


# Example usage
if __name__ == "__main__":
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
