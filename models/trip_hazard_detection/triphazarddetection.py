from ultralytics import YOLO
import os

class triphazardDetectionModel:
    """
    Handles object detection using a YOLOv8 model.
    """

    def __init__(self, main_config):
        """
        Initializes the trip hazard detection model.

        """
        self.model = YOLO(
            os.path.join(
                main_config["modelsDir"], main_config["triphazardDetectionModel"]["modelName"]
            )
        )
        self.confidence = main_config["triphazardDetectionModel"]["confidence"]
        self.imageSize = main_config["triphazardDetectionModel"]["imageSize"]
        self.device = main_config["triphazardDetectionModel"]["device"]
        self.iou = main_config["triphazardDetectionModel"]["iou"]
        self.originalClassList = main_config["triphazardDetectionModel"]["originalClassList"]
        self.detection_results = []


    def run_inference(self, image):
        """
        Inference object detection model.

        """
        results = self.model(
            image,
            imgsz=self.imageSize,
            conf=self.confidence,
            verbose=False
        )
        return results[0]
        
    
    def extract_result(self, result):
        """
        Extracts and processes detection results from the inference output.

        """
        for obj in result:
            xmin, ymin, xmax, ymax, conf, cls_id = obj.boxes.data.tolist()[0]
            self.detection_results.append([xmin, ymin, xmax, ymax, conf, cls_id])

    def __call__(self, frame):
        """
        Callable method to perform object detection on a frame.
        """
        result = self.run_inference(frame)
        self.extract_result(result)
