import os
import cv2
from PIL import Image
from ultralytics import YOLO

class ppeDetectionModel:
    """
    This class will handle ppe detection .
    First it will take frames along with person_bbox (PersonDetectionModel-->get_bbox_track_id_conf),
    crop person bbox and run inference on batched images.
    Batch size for ppe inference  will come from main config file

    Attributes:
        model (YOLO): The YOLO model instance used for ppe detection.
        confidence (float): Detection confidence threshold.
        imageSize (int): Input image size for inference.
        device (str): Device to use for inference. One of: `'cpu' or 'cuda'`
        iou (float): IOU (Intersection over Union) threshold for post-processing.
        batchSize (int): Batch size for batched inference. 
        Please be careful while selecting batch size as if it is too high, your code might crash
        predictionClasses (list): List of classes to be predicted by the model.
        personCrops (list): List to store cropped person images based on bounding boxes.
        croppedPpeBboxList (list): PPe predictions from cropped co-ords
        finalPpeBboxList (list): PPe predictions translated to original image co-ords

    """

    def __init__(self, main_config, camera_config):
        """
        constructor for yolo parameters. It takes the main config file
        and takes infos from field "ppeDetectionModel".
        Please note device is a bit complicated and if we have higher number of frames
        to process together this will be handled with "run_inference" function call only.

        Args:
            main_config (dict): Config contents read though config.json which is main config
        """
        self.ppe_model = YOLO(
            os.path.join(
                main_config["modelsDir"], main_config["ppeDetectionModel"]["modelName"]
            )
        )
        self.main_config =  main_config
        self.ppe_confidence = main_config["ppeDetectionModel"]["confidence"]
        self.ppe_imageSize = main_config["ppeDetectionModel"]["imageSize"]
        self.ppe_device = main_config["ppeDetectionModel"]["device"]
        self.ppe_iou = main_config["ppeDetectionModel"]["iou"]
        self.batchSize = main_config["ppeDetectionModel"]["batchSize"]

        self.bp_model = YOLO(
            os.path.join(
                main_config["modelsDir"],main_config["bodyPartDetectionModel"]["modelName"]
                )
            )
        self.bp_confidence = main_config["bodyPartDetectionModel"]["confidence"]
        self.bp_imageSize = main_config["bodyPartDetectionModel"]["imageSize"]
        self.bp_device = main_config["bodyPartDetectionModel"]["device"]
        self.bp_iou = main_config["bodyPartDetectionModel"]["iou"]

        self.personCrops = (
            list()
        )  # store cropped person image into this list for batched inference
        self.croppedPpeBboxList = (
            list()
        )  # Store the results of ppe model's prediction(list of list)
        self.finalPpeBboxList = (
            list()
        )  # Store the ppe results of ppe model's prediction but in full image co-ordinates
        self.croppedBpBboxList= (
            list()
        )  # Store the results of bodypart model's prediction(list of list)
        self.finalBpBboxList = (
            list()
        )  # Store the ppe results of ppe model's prediction but in full image co-ordinates
        self.validatedPpeResults = {}

    def run_ppe_inference(
        self,
        image,
    ):
        """This function will run ppe inference on the image.

        Args:
            image (np.array): Array of images 
        """
        results = self.ppe_model(
            image,
            conf=self.ppe_confidence,
            iou=self.ppe_iou,
            imgsz=self.ppe_imageSize,
            verbose=False
            # show_boxes = self.showBoxes,
        )
        return results
    
    def run_bp_inference(
            self,
            image
    ):
        """
        This method runs bodypart inference on the cropped image
        Args:
            image (np.array): Array of images 
        """
        results = self.bp_model(
        image,
        conf=self.bp_confidence,
        iou=self.bp_iou,
        imgsz=self.bp_imageSize,
        verbose=False
        # show_boxes = self.showBoxes,
        )
        return results

    def get_bbox_in_crop_img(self, results, croppedBboxList):
        """Extract the detection bbox information from the all results

        Args:
            results (yolo object): Results that came out after inference for ppe or bodypart detections
        """
        # Since we did batched inference, iterate though each image results
        for i, r in enumerate(results):
            # convert boxed to numpy array
            boxes = r.boxes.cpu().numpy()
            # list to store ppe annotations
            one_person_detections = []
            # Interate through all the predictions from each image
            for j, box in enumerate(boxes.data):
                # append into the one_ppe_list
                one_person_detections.append(box.tolist())
            # Append all predictions from one image
            croppedBboxList.append(one_person_detections)

    def crop_and_infer_person_bbox(self, original_image, personBboxes):
        """Crops the person from full image, batch it and run inference for ppe and body part detection model

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
                results_ppe = self.run_ppe_inference(self.personCrops)
                results_bp = self.run_bp_inference(self.personCrops)
                # Extract ppe bbox from whole yolo results
                self.get_bbox_in_crop_img(results_ppe, self.croppedPpeBboxList)
                self.get_bbox_in_crop_img(results_bp,self.croppedBpBboxList)
                # clear the person crops list when inference is done
                self.personCrops.clear()

        # this "if condition" is used when total number of person is less than batch size
        if self.personCrops:
            # Same as above else
            results_ppe = self.run_ppe_inference(self.personCrops)
            results_bp = self.run_bp_inference(self.personCrops)
            self.get_bbox_in_crop_img(results_ppe,self.croppedPpeBboxList)
            self.get_bbox_in_crop_img(results_bp,self.croppedBpBboxList)
            self.personCrops.clear()

    def add_final_list(self,cropList,finalList,person_bbox):
        # get all ppe predictions results from one person cropped image
        for i, one_person_prediction in enumerate(cropList):
            # iterate through each ppe predictions and translate to image co-ordinates using person bbox (xmin,ymin)
            for each_prediction in one_person_prediction:
                #Check if we have any information or just an empty list
                if each_prediction:
                    finalList.append(
                        [
                            each_prediction[0] + person_bbox[i][0],
                            each_prediction[1] + person_bbox[i][1],
                            each_prediction[2] + person_bbox[i][0],
                            each_prediction[3] + person_bbox[i][1],
                            each_prediction[4],   #detection confidence
                            int(each_prediction[5]),  # detection class
                            person_bbox[i][4],  # person track id in which detection belongs
                        ]
                    )
        # Clear the list as we don't need this information now.
        cropList.clear()

    def validate_ppe(self):
        ppe_bboxes = self.finalPpeBboxList
        bp_bboxes = self.finalBpBboxList
        
        # Initialize dictionaries for tracking PPE and body parts by track ID
        ppe_dict = {}
        bodypart_dict = {}

        validationMapping = self.main_config["ppeDetectionModel"]["validationMapping"]
        # Fill dictionaries with detected PPE and body parts by track ID
        for bbox in ppe_bboxes:
            track_id = bbox[6]
            ppe_class = self.main_config["ppeDetectionModel"]["orignalClassList"][bbox[5]]
            if track_id not in ppe_dict:
                ppe_dict[track_id] = set()
            ppe_dict[track_id].add(ppe_class)

        for bbox in bp_bboxes:
            track_id = bbox[6]
            bp_class = self.main_config["bodyPartDetectionModel"]["orignalClassList"][bbox[5]]
            if track_id not in bodypart_dict:
                bodypart_dict[track_id] = set()
            bodypart_dict[track_id].add(bp_class)

        # Initialize result dictionary
        result = {}

        # Evaluate PPE presence for each track ID
        for track_id in ppe_dict.keys() | bodypart_dict.keys():
            result[track_id] = {}
            ppe_present = ppe_dict.get(track_id, set())
            bp_present = bodypart_dict.get(track_id, set())

            for ppe, bodyparts in validationMapping.items():
                # Ensure bodyparts is treated as a list for uniform handling
                bodyparts = bodyparts if isinstance(bodyparts, list) else [bodyparts]
                
                # Determine if any required bodypart(s) are missing
                bodypart_found = all(bp in bp_present for bp in bodyparts)

                if ppe in ppe_present:
                    result[track_id][ppe] = 1  # PPE is found
                elif not bodypart_found:
                    result[track_id][ppe] = -1  # Either PPE or at least one required body part is missing
                else:
                    result[track_id][ppe] = 0  # PPE is missing but all required body parts are present

        self.validatedPpeResults = result
        # Update final PPE bounding box list with validation results



    def __call__(self, person_bbox, original_image, ):
        """Takes original image along with person bbox ([[xmin,ymin,xmax,ymax, track_id,confidence,class ],[xmin,ymin,xmax,ymax, track_id,confidence,class ]])
        for ppe detection. Some clients may want different ppe to be detected which will come from config file.
            Now after getting these three infos, we will perform three operation,
            1. Crop the person bounding rectangle image,
            2. Run batched inference for ppe detection and bodypart detection models.
            3. Translate the bbox from cropped image to full image.
            4. Validate the ppe.
        Args:
            person_bbox (list of list): prediction from person detection model
            original_image (np.array): Frame taken by cv2
            
        """
        # Crop the main image and infer it through ppe detection model
        self.crop_and_infer_person_bbox(original_image, person_bbox)
        self.add_final_list(self.croppedPpeBboxList,self.finalPpeBboxList,person_bbox)
        self.add_final_list(self.croppedBpBboxList,self.finalBpBboxList,person_bbox)
        self.validate_ppe()
        self.finalBpBboxList.clear()
        self.finalPpeBboxList.clear()