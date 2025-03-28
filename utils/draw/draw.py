import cv2
import os
import numpy as np


class drawOnFrames:
    
    def __init__(self, main_config_info,camera_config_info):
        self.cameraConfig = camera_config_info
        self.globalConfig = main_config_info
        #color coding (www.colorhexa.com)
        self.personColor = (7,129,22)
        self.fallIndicationColor =(0,0,255)
        self.personViolationColor = (49,16,226)
        self.zoneColor = (12,191,89)
        self.zoneViolationColor = (62,29,239)
        self.fireColor = (0, 0, 255)  
        self.smokeColor = (0, 0, 255) 
        self.garbageColor = (255,0,0)
        self.triphazardColor = (0,255,0)
        self.spillColor = (0,0,255)
        
        self.totalPersonInFrame = 0
        self.personInZone = 0
        self.personWithPpeViolation = 0
        self.zoneShape = "rect"
        
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.thickness = 8
        self.fontScale = 2
        self.lineType = 2
        
    def __call__(self, images_list, results_list):
        
        #Make these global variable zero for every frame
        self.totalPersonInFrame = 0
        self.personWithPpeViolation = 0
        self.personInZone = 0
        for image, results in zip(images_list, results_list):
            self.totalPersonInFrame = results["personCount"]
            #draw zones on image if we are doing person count in zone
            image = self.draw_zone(image,results)
            #If no person in results, return  image and results
            if results["personResults"]:
                image = self.draw_person(image,results)
            
            # Check if fire and smoke results exist in the results
            fire_smoke = results.get("fire_and_smoke", None)
            if fire_smoke:
                image = self.draw_fire_smoke(image, fire_smoke)

            # Check if garbage detection results exist in the results
            if "garbageDetection" in results and results["garbageDetection"]:
              image = self.draw_garbage(image, results)

            #Check if trip hazard detection results exist in the results
            if "triphazardDetection" in results and results["triphazardDetection"]:
                image = self.draw_triphazard(image, results)
        
            if results["spillDetection"]:
                image = self.draw_spill(image, results)
        
        return image     
    def calculate_ppe_violation(self,ppe_results):
        """Iterate through all ppe list and see for any violations
        if ppe class name = 0, violation else,
        ppe class name = 1, no violation

        Args:
            
            ppe_results (dict): dict containing the ppe results
            Sample ppe_results
            "ppeResults": {
                "hard-hat": 1,
                "gloves": 1,
                "mask": 0,
                "glasses": 0,
                "boots": 0,
                "vest": 1,
                "ppe-suit": 0,
                "ear-protector": 0,
                "safety-harness": 0
            },
        """

        
        ppe_violation_list = []
        for ppe_classes in ppe_results:
            if ppe_results[ppe_classes]==0:
                ppe_violation_list.append(ppe_classes)
        return ppe_violation_list
    
    
    def clamp_point(self,point, max_width, max_height):
        """Clamps the point to max_width or max_height if x or y is
        crossing the image width and hight 

        Args:
            point (_type_): _description_
            max_width (_type_): _description_
            max_height (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        x, y = point
        x = min(x, max_width)
        y = min(y, max_height)
        return [x, y]
    
    def draw_zone_on_image(self, image,zone_points,zone_color,alpha=0.5):
        """Draw zone on image 

        Args:
            image (np.array): Image on which we will draw
            zone_points (list of list): list containing list of x,y cords.
            zone_color (tuple): Color for zone (self.zoneColor or self.zoneViolationColor)
            alpha (float, optional): Used to add transparency. Defaults to 0.5.

        Returns:
            image: drawn image with zones
        """
        
        drawn_image = np.copy(image)
        if zone_points:
            height, width,shape = image.shape
            clamped_points = [self.clamp_point(point, width, height) for point in zone_points]
            
            color = (zone_color[0], zone_color[1], zone_color[2], int(255 * alpha))  # Fully opaque  color
            mask = np.zeros_like(drawn_image)
            points_array = np.array(clamped_points, dtype=np.int32)
            cv2.fillPoly(
                mask, [points_array], color=color[:3]
            )  # Ignore alpha channel for drawing

            # Blend the mask with the original image using alpha blending
            drawn_image = cv2.addWeighted(drawn_image, 1.0, mask, alpha, 0)

        return drawn_image
        
    def draw_person_with_zone_information(self, image,one_person_results,already_drawn_flag):
        """Draw person with self.personViolationColor if this person is inside 
        the zone else, return the original image

        Args:
            image (np.array): _description_
            one_person_results (dict): results of one person
            already_drawn_flag (bool): Update the flag if we are drawing person here

        Returns:
            image: Drawn image or original image
        """

        if one_person_results["zoneInformation"]["withinZone"]:
            already_drawn_flag=True
            #Draw rectangle with self.personViolationColor present in zone 
            image = cv2.rectangle(
                        image,
                        (int(one_person_results["boundingBox"]["xMin"]), int(one_person_results["boundingBox"]["yMin"])),
                        (int(one_person_results["boundingBox"]["xMax"]), int(one_person_results["boundingBox"]["yMax"])),
                        color=self.personViolationColor,
                        thickness=self.thickness,
            )
                
                    
            image =  cv2.putText(image, 
                                "Zone Violation", 
                                #(100,200),
                                (int(one_person_results["boundingBox"]["xMin"]), 
                                int(one_person_results["boundingBox"]["yMin"]-30)), 
                                self.font, 
                                self.fontScale, 
                                self.personViolationColor, 
                                self.lineType)
            
        
        return image,already_drawn_flag
    
    def draw_person_with_ppe_violations(self, image,one_person_results,ppe_violations_list,already_drawn_flag):            
        """Draw person only when we see there are some elements in ppe_violations_list as
        this list contains the classes that are missing in ppe model predictions

        Args:
            image (np.array): Image on which we are drawing
            one_person_results (dict): Result of one person
            ppe_violations_list (list): List of classes that were not detected
            already_drawn_flag (bool): Used to update when person is drawn by violation color

        Returns:
            image (np.array): drawn image 
        """

        #draw only if we have person have ppe violation
        if ppe_violations_list:
            #Update the flag that person is being drawn with voilation color
            already_drawn_flag=True
            #Update the variable used to put text on image
            self.personWithPpeViolation+=1
            # image with person bbox with personViolationColor
            image = cv2.rectangle(
                        image,
                        (int(one_person_results["boundingBox"]["xMin"]), int(one_person_results["boundingBox"]["yMin"])),
                        (int(one_person_results["boundingBox"]["xMax"]), int(one_person_results["boundingBox"]["yMax"])),
                        color=self.personViolationColor,
                        thickness=self.thickness,
            )
            #Iterate through the elements from ppe violation
            for i, text in enumerate(ppe_violations_list):
                #Put text on the image
                image =  cv2.putText(image, 
                                    text, 
                                    #(100,200),
                                    (int(one_person_results["boundingBox"]["xMax"]), 
                                    int(one_person_results["boundingBox"]["yMin"] +i* 30)), 
                                    self.font, 
                                    self.fontScale, 
                                    self.personViolationColor, 
                                    self.lineType)
            #Update the flag of person is red.
            
        return image,already_drawn_flag
        
    
    def draw_zone(self, image, results,):
        """This function will handle drawing zones on image.
            Depending on person within zone, color of zone changes between
            self.zoneColor and self.zoneViolationColor
        Args:
            image (np.array): Image
            results (dict): Results dict

        Returns:
            image (np.array): Drawn zone image
        """

        #check if we have "personCountInZone" information inside of results
        if  "personCountInZone" in results: 
            #iterate through all the zones in results            
            for zones_name in results["personCountInZone"]:
                #draw zones as self.zoneViolationColor if we have  person this zone           
                if results["personCountInZone"][zones_name]>0:
                    self.personInZone+=results["personCountInZone"][zones_name]
                    image = self.draw_zone_on_image(image,self.cameraConfig["zones"][zones_name]["zonePoints"],self.zoneViolationColor )
                #draw zones as self.zoneColor if we have no person this zone        
                else:
                    image = self.draw_zone_on_image(image,self.cameraConfig["zones"][zones_name]["zonePoints"],self.zoneColor )
        return image
                
    def draw_person(self, image,results):
        """This function will handle all drawing for person.
        Depending on the violation the color of person will be 
        self.personViolationColor or self.personColor

        Args:
            image (np.array): Image on which we have to draw person
            results (dict): Results dict

        Returns:
            image (np.array): Drawn person image
        """
       
        #Iterate through results of each person
        for one_person_results in results["personResults"]:
            #flag to see if person is drawn. Please note, we don't check this flag
            #if we are drawing person with  violations. We don't care if same person is being drawn
            #twice, just that we draw person with self.personColor when a person passes through all checks
            already_drawn_flag = False
            
            #Check if we are doing ppe detection, if yes get the ist of ppe violations and draw
            if "ppeResults" in one_person_results:
                ppe_violation_list = self.calculate_ppe_violation(one_person_results["ppeResults"])
                image,already_drawn_flag = self.draw_person_with_ppe_violations(image,one_person_results,ppe_violation_list,already_drawn_flag )
                    
            #Check if we are doing person counting in zone, if yes see which person is in zone
            if "zoneInformation" in one_person_results:
                image,already_drawn_flag = self.draw_person_with_zone_information(image,one_person_results,already_drawn_flag)
                
                
            #If none above draw person with self.personColor 
            if not already_drawn_flag:
                draw_color = self.personColor
                if "fallDetected" in one_person_results:
                    fall = one_person_results["fallDetected"]
                    draw_color = self.fallIndicationColor if fall==True else self.personColor
                image = cv2.rectangle(
                    image,
                    (int(one_person_results["boundingBox"]["xMin"]), int(one_person_results["boundingBox"]["yMin"])),
                    (int(one_person_results["boundingBox"]["xMax"]), int(one_person_results["boundingBox"]["yMax"])),
                    color=draw_color,
                    thickness=self.thickness,)
        return image  


    def draw_fire_smoke(self, image, fire_and_smoke):
        """
            This function will handle drawing for fire and smoke detections.
            Depending on the type of detection, it will draw bounding boxes 
            and labels for fire and smoke with their respective colors and confidence scores.

            Args:
                image (np.array)
                fire_and_smoke (dict)

            Returns:
                image (np.array): Image with fire and smoke bounding boxes drawn.
        """

        # Get fire and smoke bounding boxes
        fire_bboxes = fire_and_smoke.get('fire', []) or []
        smoke_bboxes = fire_and_smoke.get('smoke', []) or []

        # Draw fire bounding boxes if fire is detected
        if fire_bboxes:
            for box in fire_bboxes:
                xMin, yMin, xMax, yMax, conf = map(float, box)
                xMin, yMin, xMax, yMax = int(xMin), int(yMin), int(xMax), int(yMax)
                conf = round(conf, 2)

                # Draw bounding box and label for fire
                image = cv2.rectangle(image, (xMin, yMin), (xMax, yMax), self.fireColor, 2)
                image = cv2.putText(image, f"Fire: {conf:.2f}", (xMin, yMin - 10), self.font, self.fontScale, self.fireColor, 2)
                

        # Draw smoke bounding boxes if smoke is detected
        if smoke_bboxes:
            for box in smoke_bboxes:
                xMin, yMin, xMax, yMax, conf = map(float, box)
                xMin, yMin, xMax, yMax = int(xMin), int(yMin), int(xMax), int(yMax)
                conf = round(conf, 2)

                # Draw bounding box and label for smoke
                image = cv2.rectangle(image, (xMin, yMin), (xMax, yMax), self.smokeColor, 2)
                image = cv2.putText(image, f"Smoke {conf:.2f}", (xMin, yMin - 10), self.font, self.fontScale, self.smokeColor, 2)
        return image
    
    def draw_garbage(self, image, fullImageResults):
        """
            This function will handle drawing for garbage detections.   
        """
        
        garbages = fullImageResults["garbageDetection"]["garbage"]

        for garbage in garbages:
            [xmin, ymin, xmax,ymax]= map(int,[garbage["xmin"], garbage["ymin"], garbage["xmax"], garbage["ymax"]])

            image = cv2.rectangle(image, (xmin, ymin), (xmax,ymax), self.garbageColor, 3)
            image = cv2.putText(image, f"Garbage", (xmin,ymin - 10),self.font,self.fontScale, self.garbageColor, 3)

        return image

 
    def draw_triphazard(self, image, fullImageResults):
        """
        Handles drawing for trip hazard detections.
        """
        # Draw trip zones on the image
        image = self.draw_tripzone_on_image(image, fullImageResults)

        # Iterate over each zone in the triphazardDetection results
        for zone_data in fullImageResults["triphazardDetection"].values():
            if zone_data.get("status", False):
                # Iterate over the bounding boxes for this zone
                for bbox in zone_data.get("object_bbox", []):
                    xmin, ymin, xmax, ymax = bbox
                    # Draw the bounding box for the trip hazard
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.triphazardColor, 3)
                    image = cv2.putText(image, "Object", (xmin, ymin - 10), self.font, self.fontScale, self.triphazardColor, 3)

        return image

    def draw_tripzone_on_image(self, image, results):
        """
        Draws all zones on the image. Zones are always drawn, but the color depends on 
        whether a hazard is detected.
        """
        # Iterate over all zones in the camera configuration
        for zone_key, zone_config in self.cameraConfig["tripzones"].items():
            zone_points = zone_config.get("zonePoints", [])

            if not zone_points:
                continue

            # Check if the zone has a hazard directly
            zone_id = str(zone_config["id"])  
            if results.get("triphazardDetection", {}).get(zone_id, {}).get("status", False):
                # Hazard detected, draw with violation color
                image = self.draw_zone_on_image(image, zone_points, self.zoneViolationColor)
            else:
                # No hazard, draw with regular zone color
                image = self.draw_zone_on_image(image, zone_points, self.zoneColor)

        return image


    def video_save_init(self, init_video, output_file_name, video_save_dir):

        """This will initialize the video saving part

        Args:
            init_video (cv2 object): This will help us extract relevent infos
            output_file_name (str/path): path to save the file
        """

        os.makedirs(video_save_dir, exist_ok=True)
        frame_width = int(init_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(init_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(init_video.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )  # You can also use other codecs like MJPG, X264, etc.

        video_out = cv2.VideoWriter(
            os.path.join(video_save_dir, output_file_name),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

        return video_out    


