def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in COCO format.
    Boxes are defined as [x, y, w, h].

    Parameters:
        box1: list or array, bounding box 1 [x, y, w, h]
        box2: list or array, bounding box 2 [x, y, w, h]

    Returns:
        iou: float, Intersection over Union value
    """
    # Extract coordinates and dimensions
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Area of union
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou


# The Bbox must be passed in the COCO format [x,y,w,h]
def convert_to_coco_format(pred_box):
    # the input is in format x_top_left,y_top_left,x_bottom_right,y_bottom_right and I want in format x,y,w,h
    new_box = pred_box.copy()
    new_box[2] = abs(pred_box[2] - pred_box[0])
    new_box[3] = abs(pred_box[1] - pred_box[3])
    return new_box


class IntersectionOverUnion:
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
        iou_scores = []

        for idx in range(self.dataset.__len__()):
            # Get the ground truth bounding boxes and labels
            image, ground_truth_boxes, ground_truth_labels = self.dataset.__getitem__(idx)

            # Predict using the model
            box_predicted, label_predicted = self.model(image)

            # Compare each predicted bounding box to the ground truth boxes
            for pred_box, pred_label in zip(box_predicted, label_predicted):
                # Find the matching ground truth box (if any)
                matching_boxes = [
                    calculate_iou(convert_to_coco_format(pred_box), gt_box)
                    for gt_box, gt_label in zip(ground_truth_boxes, ground_truth_labels)
                    if pred_label == gt_label
                ]

                # If there are matches, take the highest IoU
                if matching_boxes:
                    best_iou = max(matching_boxes)
                    iou_scores.append(best_iou)

        # Calculate average IoU
        average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        return average_iou
