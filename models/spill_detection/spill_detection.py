from ultralytics import YOLO
import os

class spillDetectionModel:

    def __init__(self, main_config):
        self.model = YOLO(
            os.path.join(
                main_config["modelsDir"], main_config["spillDetectionModel"]["modelName"]
            )
        )
        self.confidence = main_config["spillDetectionModel"]["confidence"]
        self.imageSize = main_config["spillDetectionModel"]["imageSize"]
        self.device = main_config["spillDetectionModel"]["device"]
        self.orignalClassList = main_config["spillDetectionModel"]["originalClassList"]
        self.spill_results = []


    def run_inference(self,image):
        results = self.model(
            image,
            imgsz=self.imageSize,
            conf = self.confidence,
            verbose=False
        )
        return results[0]
    
    def extract_result(self,result):
        for spill in result:
            xmin, ymin, xmax, ymax, conf, cls_id = spill.boxes.data.tolist()[0]
            self.spill_results.append([xmin,ymin,xmax, ymax, conf])

    def __call__(self,frame):
        
        result = self.run_inference(frame)
        self.extract_result(result)
