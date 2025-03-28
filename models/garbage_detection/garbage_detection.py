from ultralytics import YOLO
import os

class garbageDetectionModel:
    """
    Handles fire and smoke detection using a YOLOv8 model.
    """

    def __init__(self, main_config):
        """
        Initializes the garbage detection model.

        """
        self.model = YOLO(
            os.path.join(
                main_config["modelsDir"], main_config["garbageDetectionModel"]["modelName"]
            )
        )
        self.confidence = main_config["garbageDetectionModel"]["confidence"]
        self.imageSize = main_config["garbageDetectionModel"]["imageSize"]
        self.device = main_config["garbageDetectionModel"]["device"]
        self.iou = main_config["garbageDetectionModel"]["iou"]
        self.orignalClassList = main_config["garbageDetectionModel"]["originalClassList"]
        self.garbage_results = []


    def run_inference(self,image):
        """
        Inference garbage detection model.

        """
        results = self.model(
            image,
            imgsz=self.imageSize,
            conf = self.confidence,
            verbose=False
        )
        return results[0]
        
    
    def extract_result(self,result):
        """
        Extracts and processes detection results from the inference output.

        """
        for garbage in result:
            xmin, ymin, xmax, ymax, conf, cls_id = garbage.boxes.data.tolist()[0]
            self.garbage_results.append([xmin,ymin,xmax, ymax, conf])

    def __call__(self,frame):
        """
        Callable method to perform garbage detection on a frame.
        """
        result = self.run_inference(frame)
        self.extract_result(result)
        
