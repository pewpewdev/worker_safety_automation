import logging
import os
import cv2

from ultralytics import YOLO


class fallDetectionModel:
    """
    This class will handle fall classification .
    First it will take frames along with person_bbox, crop 
    images and run inference on person crops

    Attributes:
        model (YOLO): The YOLO model instance used for fall ckassification.
        confidence (float): Detection confidence threshold.
        imageSize (int): Input image size for inference.
        device (str): Device to use for inference. One of: `'cpu' or 'cuda'`
        originalClassList: List of classes (in this case fall and notfall)
        fall_result : results obtained from the fall classifier
    """

    def __init__(self, main_config):
        self.model = YOLO(
            os.path.join(
                main_config["modelsDir"], main_config["fallDetectionModel"]["modelName"]
            )
        )
        self.fall_confidence = main_config["fallDetectionModel"]["fall_confidence"]
        self.imageSize = main_config["fallDetectionModel"]["imageSize"]
        self.device = main_config["fallDetectionModel"]["device"]
        self.orignalClassList = main_config["fallDetectionModel"]["originalClassList"]
        self.fall_result= (list())
        self.batchSize = main_config["fallDetectionModel"]["batchSize"]
        self.personCrops = (
            list()
        )  # store cropped person image into this list for batched inference

    def run_inference(self,image):
        results = self.model(
            image,
            imgsz=self.imageSize,
            classes=self.orignalClassList,
            verbose=False
        )
        return results
    
    def validate_fall_conf(self, fall_result,person_bbox):
        """
        If fall detected, model sends output as 0
        If no fall detected, model sends output as 1
        """
        validated_res = []
        for i in range(len(fall_result)):
            if fall_result[i].probs.top1 == 0:
                if fall_result[i].probs.top1conf > self.fall_confidence:
                    validated_res.append([1,person_bbox[i][4]])
                else:
                    validated_res.append([0,person_bbox[i][4]])
            else:
                validated_res.append([0,person_bbox[i][4]])
        return validated_res
    
    def final_fall_result(self,result, person_bbox):
        results= []
        for i in range(len(result)):
            results.append([result[i],person_bbox[i][4]])
        return results
            
    def crop_and_infer_person_bbox(self, original_image, personBboxes):
        """Crops the person from full image and run inference for fall classification model

        Args:
            original_image (np.array): full frame
            personBboxes (list of list): prediction from person detection model
        """
        # Go through each person predictions
        for bbox in personBboxes:
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            #Check if we can add more crops in self.personCrops list
            if len(self.personCrops) < self.batchSize:
                # crop and append if there is scope of appending images inside personCrops list
                self.personCrops.append(original_image[ymin:ymax, xmin:xmax])
            else:
                # if len of personCrops meet batch size, run inference
                results = self.run_inference(self.personCrops)
                self.fall_result = self.validate_fall_conf(results,personBboxes)
                self.personCrops.clear()
        
        if self.personCrops:
            # Same as above else
            results = self.run_inference(self.personCrops)
            self.fall_result = self.validate_fall_conf(results, personBboxes)
            self.personCrops.clear()
    
    def __call__(self, frame, person_bboxes):
        self.crop_and_infer_person_bbox(frame, person_bboxes)




    