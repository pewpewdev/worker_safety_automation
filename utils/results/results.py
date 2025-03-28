import logging

from shapely.geometry import Point, Polygon
import json

# Configure logging
logging.basicConfig(level=logging.WARNING)


class jsonResultsManager:
    """This class is responsible for adding results in json and calculating
    the person within the zone

    """

    def __init__(
        self,
        camera_config,
    ):
        """Constructor for defining the schema for results

        Args:
            camera_config (dict): Camera cofig
        """
        self.cameraConfig = camera_config
        self.camId = camera_config["camID"]
        self.description = camera_config["description"]
        self.fullImageResults = None
        self.onePersonResults = None
        
        
    def init_template(self, camera_config):
        """Initialize the templates for fullImageResults and onePersonResults

        Args:
            camera_config (dict): Video config file
        """
        self.fullImageResults = self.define_full_image_results_schema(camera_config)
        self.onePersonResults = self.define_single_person_results_schema(camera_config)
        
        
    def define_full_image_results_schema(self, camera_config):
        """This function will define the results schema for full image
        This will be sent to our next stage
        Sample:
        {
            "camId": "Heavy_vehicle_backing_near_worker.mov",
            "description": "give desciption about camera",
            "frameID": 455,
            "personCount": 2,
            "personResults": [],
            "fallDetected": 0
            "personCountInZone": {
                "zone1": 1,
                "zone2": 0
            }
        }

        """
        fullImageResults = dict()
        # camera id comes from camera json
        fullImageResults["camId"] = self.camId
        # Description about the camera comes from  camera json
        fullImageResults["description"] = self.description
        # Frame id to keep track
        fullImageResults["frameID"] = 0
        # Count of person in current frame
        fullImageResults["personCount"] = 0
        # Individual person results
        fullImageResults["personResults"] = []      
          
        if camera_config["analytics"]["fallDetection"]:
            fullImageResults["fallDetected"] = False
            
        if camera_config["analytics"].get("fire_smoke_detection", False):
            # define dict for fire and smoke detection results
            fullImageResults["fire_and_smoke"] = {
                "fire": [],
                "smoke": [],
                "fire_detected": False,
                "smoke_detected": False,
            }
                
        if camera_config["analytics"]["garbageDetection"]:
            # define dict for garbage detection results
            fullImageResults["garbageDetection"] = {
                "garbage":[],
                "garbage_detected":False
            }

        if camera_config["analytics"]["tripHazardDetection"]:
            # define dict for trip hazard detection results
            fullImageResults["triphazardDetection"] = dict()
            # Iterate through all zones in camera config
            for each_zone in camera_config["tripzones"]:
                zone_id = camera_config["tripzones"][each_zone]["id"]
                # Initialize the structure for the current zone with status and empty bounding boxes
                fullImageResults["triphazardDetection"][str(zone_id)] = {
                    "status": False,
                    "object_bbox": []
                }        if camera_config["analytics"]["spillDetection"]:
            fullImageResults["spillDetection"] = {
                "spill":[],
                "spill_detected":False
            }



        # Add results template if working with person in zone counting
        if camera_config["analytics"]["personInZoneCounting"]:
            # define dict for full results
            fullImageResults["personCountInZone"] = dict()
            # Iterate through all zones in camera config
            for each_zone in camera_config["zones"]:
                # Get zone name
                zone_name = camera_config["zones"][each_zone]["name"]
                # add zone name in camera config
                fullImageResults["personCountInZone"][zone_name] = 0
        return fullImageResults

    def define_single_person_results_schema(self, camera_config):
        """Create results schema for one person. Please not the results
        template will change based on analytics you are performing
        Sample:
                {
            "personId": 1,
            "boundingBox": {
                "xMin": 1611,
                "yMin": 353,
                "xMax": 2102,
                "yMax": 1089
            },
            "centroid": {
                "x": 1856,
                "y": 721
            },
            "fallDetected":0
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
            "zoneInformation": {
                "withinZone": false,
                "zoneName": null,
                "zoneID": null
            },
            
        }

        Args:
            camera_config (dict): camera configuration
        """
        one_person_results = dict()
        only_person_results = {
            "personId": 0,
            "boundingBox": {"xMin": 0, "yMin": 0, "xMax": 0, "yMax": 0},
            "centroid": {"x": 0, "y": 0},
        }
        one_person_results.update(only_person_results)

        # Add fall detection if we are working with fall detection
        if camera_config["analytics"]["fallDetection"]:
            one_person_results["fallDetected"] = False
            
        # Add zone information in person results if we are working person within the zone
        if camera_config["analytics"]["personInZoneCounting"]:
            zone_results = {
                "zoneInformation": {
                    "withinZone": False,
                    "zoneName": None,
                    "zoneID": None,
                }
            }

            one_person_results.update(zone_results)

        # Add ppe information in person results if we are working with ppe detection
        # Please note we may not want all ppes to be detected so will have to be filtered
        # Camera config of contains for which ppe we want to detect
        if camera_config["analytics"]["ppeDetection"]:
            ppe_dict = {"ppeResults": {}}
            
            for items,status in camera_config["ppeDetection"].items():
                
                if status:
                    ppe_dict["ppeResults"][items] = False
            if not ppe_dict:
                logging.warning(
                    "You have marked analytics-->ppeDetection as true but all ppes from ppeDetection are false. Please double check"
                )

            one_person_results.update(ppe_dict)
        return one_person_results

    def add_person_results(self, person_bboxes, frame_id):
        """Append the person results from predictions to json

        Args:
            person_bboxes (list of lists): Person predictions from person detection model
            frame_id (int): frame id
        """
        
        # Iterate through each person prediction
        for person_bbox in person_bboxes:
            # APPEnd and update the results
            self.onePersonResults["personId"] = int(person_bbox[4])
            self.onePersonResults["boundingBox"]["xMin"] = person_bbox[0]
            self.onePersonResults["boundingBox"]["yMin"] = person_bbox[1]
            self.onePersonResults["boundingBox"]["xMax"] = person_bbox[2]
            self.onePersonResults["boundingBox"]["yMax"] = person_bbox[3]
            self.onePersonResults["centroid"]["x"] = int(
                (person_bbox[0] + person_bbox[2]) / 2
            )
            self.onePersonResults["centroid"]["y"] = int(
                (person_bbox[1] + person_bbox[3]) / 2
            )

            self.fullImageResults["personCount"] += 1
            self.fullImageResults["personResults"].append(self.onePersonResults)
            # Re-init the one person template to add new person's info
            self.onePersonResults = self.define_single_person_results_schema(self.cameraConfig)
            
        self.fullImageResults["frameID"] = frame_id

    def add_ppe_results(self, validatedPpeDict):
        
        """Add ppe results into fullImageResults["personResults"]

        Args:
            validatedPpeDict (list ): dictionary for ppe validated results
        """
        
        # Iterate through all the results of person inside "fullImageResults"
        for person_dict in self.fullImageResults["personResults"]:
            #iterate through all ppes predictions
            for track_id,results in validatedPpeDict.items():
                # Check if person id is matching both in ppe and person
                if track_id== person_dict["personId"]:
                    person_dict["ppeResults"] = results

    def add_fall_results(self,fall_results):
        #fall_status  format [fall,track_id]
        for fall_result in fall_results:
            for person_dict in self.fullImageResults["personResults"]:
                if fall_result[1] == person_dict["personId"]:
                    person_dict["fallDetected"] = bool(fall_result[0])
                    if fall_result[0]:
                        self.fullImageResults["fallDetected"] = True

    def add_fire_smoke_results(self, fire_bboxes, smoke_bboxes,fire_flag, smoke_flag):
        """
        Adds fire and smoke detection results to fullImageResults.

        Args:
            fire_bboxes (list): List of fire bounding boxes 
            smoke_bboxes (list): List of smoke bounding boxes
 
        """

        # Add fire bounding boxes if provided
        if fire_bboxes:
            self.fullImageResults["fire_and_smoke"]["fire"] = fire_bboxes  
            self.fullImageResults["fire_and_smoke"]["fire_detected"] = True  
        else:
            self.fullImageResults["fire_and_smoke"]["fire_detected"] = False  

        # Add smoke bounding boxes if provided
        if smoke_bboxes:
            self.fullImageResults["fire_and_smoke"]["smoke"] = smoke_bboxes  
            self.fullImageResults["fire_and_smoke"]["smoke_detected"] = True  
        else:
            self.fullImageResults["fire_and_smoke"]["smoke_detected"] = False 

    def add_spill_results(self,spill_results):
        if len(spill_results) != 0:
            spill_dict = {}
            self.fullImageResults["spillDetection"]["spill_detected"] = True
            for spill in spill_results:

                xmin,ymin,xmax,ymax,conf = spill
                spill_dict["xmin"] = int(xmin)
                spill_dict["ymin"] = int(ymin)
                spill_dict["xmax"] = int(xmax)
                spill_dict["ymax"] = int(ymax)
                
                self.fullImageResults["spillDetection"]["spill"].append(spill_dict)


            print("Full Image Results: ", json.dumps(self.fullImageResults))
            
    def add_garbage_results(self,garbage_results):
            """
            Adds garbage results to fullImageResults.
            
            """
            if len(garbage_results) != 0:
                garbage_dict = {}
                self.fullImageResults["garbageDetection"]["garbage_detected"] = True
                for garbage in garbage_results:

                    xmin,ymin,xmax,ymax,conf = garbage
                    garbage_dict["xmin"] = int(xmin)
                    garbage_dict["ymin"] = int(ymin)
                    garbage_dict["xmax"] = int(xmax)
                    garbage_dict["ymax"] = int(ymax)
                    
                    self.fullImageResults["garbageDetection"]["garbage"].append(garbage_dict)


    
    def add_triphazard_results(self, detection_results):
        """
        Adds trip hazard results to fullImageResults, filtering only those detected
        objects whose center points fall within the trip zones.
        """

        # Iterate through each detection result
        for triphazard in detection_results:
            xmin, ymin, xmax, ymax, conf, _ = triphazard

            # Calculate the center point of the bounding box
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            center_point = Point(center_x, center_y)

            # Iterate through each trip zone directly from the cameraConfig
            for zone_data in self.cameraConfig["tripzones"].values():
                zone_points = zone_data.get("zonePoints", [])
                zone_id = zone_data.get("id", None)

                # if zone points are valid and create a polygon
                if zone_points:
                    zone_polygon = Polygon(zone_points)

                    # if the center point is inside the zone
                    if center_point.within(zone_polygon):
                        # Add the bounding box to the zone's bbox list
                        self.fullImageResults["triphazardDetection"][str(zone_id)]["object_bbox"].append([
                            int(xmin),
                            int(ymin),
                            int(xmax),
                            int(ymax)
                        ])

                        # Update the zone's status to True since we have detected a hazard
                        self.fullImageResults["triphazardDetection"][str(zone_id)]["status"] = True
                        break  
                # else:
                #     logging.warning(f"No zone points found for zone ID {zone_id}")

