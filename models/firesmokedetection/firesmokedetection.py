import os
from ultralytics import YOLO



class FireSmokeDetectionModel:
    """
    Handles fire and smoke detection using a YOLOv8 model.
    Dynamically fetches detection flags and class IDs from camera and global configs.

    Attributes:
        fire_bboxes (list): Stores bounding boxes for detected fire instances.
        smoke_bboxes (list): Stores bounding boxes for detected smoke instances.
    """

    def __init__(self, main_config):
        """
        Constructor for YOLO parameters. Takes main and camera config files.

        Args:
            main_config (dict): Global configuration (e.g., model paths, parameters).
            camera_config (dict): Camera-specific configuration (e.g., analytics flags).
        """
        
        
        # Load YOLO model
        self.model = YOLO(
            os.path.join(
                main_config["modelsDir"], main_config["FireSmokeDetectionModel"]["modelName"]
            )
        )

        # YOLO parameters
        self.conf = main_config["FireSmokeDetectionModel"]["confidence"]
        self.imageSize = main_config["FireSmokeDetectionModel"]["imageSize"]
        self.device = main_config["FireSmokeDetectionModel"]["device"]
        self.iou = main_config["FireSmokeDetectionModel"]["iou"]
        self.batchSize = main_config["FireSmokeDetectionModel"]["batchSize"]
        self.predictionClasses = main_config["FireSmokeDetectionModel"]["predictionClasses"]
        self.originalClassList = main_config["FireSmokeDetectionModel"]["orignalClassList"]
        self.fire_class_id = self.predictionClasses[self.originalClassList.index("fire")]
        self.smoke_class_id = self.predictionClasses[self.originalClassList.index("smoke")]
        self.showBoxes = main_config["FireSmokeDetectionModel"]["showBoxes"]
        # Initialize bounding box storage
        self.fire_bboxes = []
        self.smoke_bboxes = []
        self.fire_detected = False
        self.smoke_detected = False

    def get_bbox_class_conf(self, results):
        """
        Extract bounding boxes for fire and smoke classes from YOLO results.

        Args:
            results (YOLO object): YOLO detection results.

        Populates:
            self.fire_bboxes (list): Bounding boxes for detected fire.
            self.smoke_bboxes (list): Bounding boxes for detected smoke.
        """
        for r in results:
            boxes = r.boxes.data.cpu().numpy()
            for box in boxes:
                xmin, ymin, xmax, ymax, conf, class_id = box[:6]
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                conf = float(conf)
                class_id = int(class_id)

                # Process fire detections
                if class_id == self.fire_class_id:
                    self.fire_bboxes.append([xmin, ymin, xmax, ymax, conf])
                    self.fire_detected = True
                # Process smoke detections
                elif class_id == self.smoke_class_id:
                    self.smoke_bboxes.append([xmin, ymin, xmax, ymax, conf])
                    self.smoke_detected = True

    def __call__(self, image):
        """
        Runs inference on the input image.

        Args:
            image (np.array): Input image or batch of images.

        Returns:
            dict or None: Result dictionary containing bounding boxes and detection flags,
                          or None if detection is disabled.
        """
        
        #reinitialize the flags and bbox list
        self.fire_bboxes.clear()
        self.smoke_bboxes.clear()
        self.fire_detected = False
        self.smoke_detected = False
        # Perform detection
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imageSize,
            classes=self.predictionClasses,
            device=self.device,
            verbose=False,
        )
        

        # Populate bounding box lists
        self.get_bbox_class_conf(results)
