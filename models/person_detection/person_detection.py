import logging
import os

import cv2
from ultralytics import YOLO


class personDetectionModel:
    """This class will handle person detection and tracking
    using yolov8 model


    Attributes:
        model (YOLO): The YOLO model instance used for detection.
        confidence (float): Detection confidence threshold.
        imageSize (int): Input image size for inference.
        device (str): Device to use for inference (e.g., 'cpu', 'cuda').
        iou (float): IOU (Intersection over Union) threshold for post-processing.
        batchSize (int): Batch size for batched inference (currently not utilized).
        predictionClasses (list): List of classes to be predicted by the model.
        showBoxes (bool): Whether to display bounding boxes on detected objects.
        personBboxes (list): List to store detected person bounding boxes.

        Methods:
        get_bbox_track_id_conf(): extracts predictions results from predictd yolo results
        crop_person_boxes: Extract person crops from full image
    """

    def __init__(self, config):
        """
        constructor for yolo parameters. It takes the main config file as input
        and extract relevent infos from field "PersonDetectionModel".

        Args:
            config (dict): Config contents read though config.json which is main config
        """
        self.model = YOLO(
            os.path.join(
                config["modelsDir"], config["PersonDetectionModel"]["modelName"]
            )
        )
        self.confidence = config["PersonDetectionModel"]["confidence"]
        self.imageSize = config["PersonDetectionModel"]["imageSize"]
        self.device = config["PersonDetectionModel"]["device"]
        self.iou = config["PersonDetectionModel"]["iou"]
        self.batchSize = config["PersonDetectionModel"]["batchSize"]
        self.predictionClasses = config["PersonDetectionModel"]["predictionClasses"]
        self.showBoxes = config["PersonDetectionModel"]["showBoxes"]
        self.personBboxes = list()

    def get_bbox_track_id_conf(self, results):
        """This function will take person detection results as input generate a list of lists.
        Each list will contain seven infos,
        index 0: xmin
        index 1: ymin
        index 2: xmax
        index 3: ymax
        index 4: track_id
        index 5: confidence class
        index 6: class

        Args:
            results (yolo object):
        """
        for i, r in enumerate(results):
            boxes = r.boxes.cpu().numpy()
            for j, box in enumerate(boxes):
                try:
                    xmin = int(box.data[0][0])
                    ymin = int(box.data[0][1])
                    xmax = int(box.data[0][2])
                    ymax = int(box.data[0][3])
                    track_id = box.data[0][4]
                    conf = box.data[0][5]
                    class_id = box.data[0][6] 
                    #append these in self.personBboxes list
                    self.personBboxes.append(
                        [xmin, ymin, xmax, ymax, track_id, conf, class_id]
                    )

                except Exception as e:
                    logging.warning(
                        "expected total length of box to be six but track_id missing, skipping frame"
                    )
                    continue

    def __call__(self, image,):
        """
        Run inference on the input image.

        Args:
            image (np.array): Input image or batch of images.
        """
        # Perform detection and tracking using YOLO
        # try:
        results = self.model.track(
            image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imageSize,
            classes=self.predictionClasses,
            show_boxes=self.showBoxes,
            persist=True,
            verbose=False,
        )
        # Extract bounding boxes, class id, track_id and confidence
        self.get_bbox_track_id_conf(results)

