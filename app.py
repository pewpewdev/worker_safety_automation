import logging
import os
import glob
import cv2
import numpy
import json

import argparse

from models import (personCountInZone, personDetectionModel, ppeDetectionModel,FireSmokeDetectionModel,fallDetectionModel,garbageDetectionModel,triphazardDetectionModel,spillDetectionModel,
                    reID)
from utils import drawOnFrames, jsonConfigParser, jsonResultsManager, S3VideoDownloader


logging.basicConfig(level=logging.INFO)


class VideoProcessor:
    """
    A class to process videos using various models and configurations.
    Please note two config files are needed one camera config file and
    main config file.

    Attributes:git checkout -b fire_and_smoke

        globalConfigInfo: Main config file for our pipeline. Remains same for every video
        personDetectionPipeline: Handles person detection on full images.
        cameraConfigInfo: config files for each video
        ppeDetectionPipeline: Handles ppe detection on cropped images
        personInZoneCounting: handles counting person in a particular zone
        fireSmokeDetectionPipeline: Handles fire and smoke  on full image
        reidPipeline: Handles the re-identification pipeline
        videosDir: Path to local videos dir
        jsonResultsManager: Defines results schema and add different results values in
                            respective fields
        drawOnFrames: Handles drawing different results based on compliance

    Methods:
        __call_: Call method to run our class as function
        init_pipelines: Initializes different pipelines based on camera.
                eg: one camera may only need person in zone counting so we define those
                in that respective cam json
        get_camera_config_info: this will read the camera config information
        process_video: Start processing the individual videos from __call__
        process_frame: process each frame via different pipelines via "process_video" method
        process_ppe_detection: run ppe detection on frame.
        process_zone_counting: Counting the number of people in zone.
        process_fire_and_smoke: Run fire and smoke detection on frame.
        process_garbage_detection: Run garbage detection on frame

    Order of Execution:
    1. __call__
    2. init_pipelines
    3. process_video
    4. process_frame
    5. process_ppe_detection
    6. process_zone_counting
    7. process_fire_and_smoke
    8. process_garbage_detection
    9. process_triphazard_detection
    """

    def __init__(self, config_path):
        """FallDetector
        Initialize the VideoProcessor object.

        Args:
            config_path (str): Path to the main configuration file.
        """
        #delete old db files if exists
        self.delete_old_db_files()
        # Load main config file, this is our global config file.
        self.globalConfigInfo = jsonConfigParser(config_path).config

        # Initialize different piplelines with None values
        self.personDetectionPipeline = None
        self.cameraConfigInfo = None
        self.ppeDetectionPipeline = None
        self.personInZoneCounting = None
        self.fireSmokeDetectionPipeline = None
        self.garbageDetectionPipeline = None
        self.triphazardDetectionPipeline = None
        self.jsonResuljsonResultsManagertsManager = None
        self.drawOnFrames = None
        self.reidPipeline = None   
        self.fallDetectionPipeline = None
        # Path to all videos directory
        self.videosDir = self.globalConfigInfo["videoDownloader"]["localVideoPath"]
        
        #download and videos if not already downloaded
        #S3VideoDownloader(self.globalConfigInfo) 
        
    def delete_old_db_files(self):
        """This method deletes the old database files (.db) from the current directory.
        This will remain only if script exectuion is stopped in between. If not, this is handled in
        self.process_video method
        """
        db_files = glob.glob("*.db")

        # Iterate over the list of files and remove each one
        for db_file in db_files:
            os.remove(db_file)
            print(f"Deleted: {db_file}")
            
    def __call__(self):
        """
        Process all videos in the specified directory. This reads the video path and
        use process_video to process individual videos
        """
        video_files_list = os.listdir(self.videosDir)
        if not video_files_list:
            logging.error("Video dir is empty  %s", self.videosDir)

        for video_file in video_files_list:
            print("processing video  %s", video_file)
            # Initialize camera specific pipelines
            self.init_pipelines(video_file)
            # process videos
            self.process_video(os.path.join(self.videosDir, video_file), video_file)
            
            

    def init_pipelines(self, video_file):
        """Initializes various other pipelines based on camera

        Args:
            video_file (str): Name of video file
        """
        #stash all pipelines from previous runs
        self.personDetectionPipeline=None
        self.ppeDetectionPipeline = None
        self.fallDetectionPipeline = None
        self.personInZoneCounting = None
        self.jsonResultsManager = None
        self.reidPipeline = None
        self.fireSmokeDetectionPipeline = None
        self.garbageDetectionPipeline = None
        self.triphazardDetectionPipeline = None
        self.spillDetectionPipeline = None
        self.personDetectionPipeline = personDetectionModel(self.globalConfigInfo)
        # read camera config, if not found throw error
        self.cameraConfigInfo = self.get_camera_config_info(video_file)
        if not self.cameraConfigInfo:
            logging.error(
                "An error occurred, config file not found for  %s", video_file
            )
        #Initialize the drawing on frame pipeline    
        self.drawOnFrames = drawOnFrames(self.globalConfigInfo,self.cameraConfigInfo)
        # Initialize the reid pipeline
        self.reidPipeline = reID(self.cameraConfigInfo, self.globalConfigInfo)

        # Initialize the json results manager
        self.jsonResultsManager = jsonResultsManager(self.cameraConfigInfo)

        # Init ppe detection pipeline
        if self.cameraConfigInfo["analytics"]["ppeDetection"]:
            self.ppeDetectionPipeline = ppeDetectionModel(
                self.globalConfigInfo, self.cameraConfigInfo
            )
        
        #initialize fall detection pipeline
        if self.cameraConfigInfo["analytics"]["fallDetection"]:
            self.fallDetectionPipeline = fallDetectionModel(self.globalConfigInfo)

        #initialize garbage detection pipeline

        if self.cameraConfigInfo["analytics"]["garbageDetection"]:
            self.garbageDetectionPipeline = garbageDetectionModel(self.globalConfigInfo)

        #initialize trip hazard detection pipeline
        if self.cameraConfigInfo["analytics"]["tripHazardDetection"]:
            self.triphazardDetectionPipeline = triphazardDetectionModel(self.globalConfigInfo)

        #initialize spill detection pipeline

        if self.cameraConfigInfo["analytics"]["spillDetection"]:
            self.spillDetectionPipeline = spillDetectionModel(self.globalConfigInfo)

        # Init person in zone counting
        if self.cameraConfigInfo["analytics"]["personInZoneCounting"]:
            self.personInZoneCounting = personCountInZone(self.cameraConfigInfo)
            if not self.personInZoneCounting.validZones:
                logging.warning(
                    "All zone points are empty, skipping counting person in zone"
                )

                self.personInZoneCounting = None
            
        # Initialize fire and smoke detection pipeline if enabled in camera config        
        fireandsmoke = self.cameraConfigInfo["analytics"]["fire_smoke_detection"]
        
        if fireandsmoke:
            self.fireSmokeDetectionPipeline = FireSmokeDetectionModel(self.globalConfigInfo)
            

                
    def get_camera_config_info(self, video_path):
        """
        Get camera configuration information from the corresponding JSON file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict: Camera configuration information.
        """
        camera_config_file = os.path.join(
            "config", os.path.splitext(video_path)[0] + ".json"
        )

        return jsonConfigParser(camera_config_file).config
    
    def process_video(self, video_path, video_file_name):
        """
        Process an individual video file.

        Args:
            video_path (str): Path to the video file.
            video_file_name(str): Name of video we are processing
        """

        cap = cv2.VideoCapture(video_path)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        frame_id = 0
        # initialize video writer
        video_out_file = self.drawOnFrames.video_save_init(
            cap, video_file_name, self.globalConfigInfo["videoSaveDir"]
        )

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                #check if the frame is valid, opencv used numpy arry to store the images
                if not isinstance(frame, numpy.ndarray):
                    logging.warning("error with the frame")
                    continue
                drawn_frame = self.process_frame(frame, frame_id, video_file_name)
                # save the frame in video
                video_out_file.write(drawn_frame)
                frame_id += 1
                cv2.imshow("frame", drawn_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break                
            
        cap.release()
        video_out_file.release()
        # remove the local db file

        os.remove(self.reidPipeline.local_database_name)
        self.reidPipeline.tracklets.clear()

    

    def process_frame(self, frame, frame_id, video_name):
        """Main function for running different pipelines on frame

        Args:
            frame (np.array): Opencv read frame
            frame_id (int): frame id
            video_name (str): Name of the video file

        Returns:
            np.array: drawn image for different piplelines
        """
        #initialize results template:
        self.jsonResultsManager.init_template(self.cameraConfigInfo)

        # Run fire and smoke detection pipeline
        if self.fireSmokeDetectionPipeline:
            self.process_fire_and_smoke(frame)
            
        
        #Run Person detection
        self.personDetectionPipeline(frame, )
        self.reidPipeline(self.personDetectionPipeline.personBboxes, frame)

        frameLevelInference = self.fireSmokeDetectionPipeline or self.garbageDetectionPipeline or self.triphazardDetectionPipeline

        if not self.personDetectionPipeline.personBboxes and not frameLevelInference:
            return frame
        
        # add person results in json results, mainly bbox, and track_id
        if self.personDetectionPipeline.personBboxes:
            #print("Adding person results!")
            self.jsonResultsManager.add_person_results(
                self.personDetectionPipeline.personBboxes, frame_id
            )

        # Run ppe detection pipeline on image
        if self.ppeDetectionPipeline:
            self.process_ppe_detection(frame, self.personDetectionPipeline.personBboxes)

        # Run fall detection pipeline on image
        if self.fallDetectionPipeline:
            self.process_fall_detection(frame, self.personDetectionPipeline.personBboxes)
            #clear data structures after adding to results
            self.fallDetectionPipeline.fall_result.clear()

        # Run garbage detection pipeline on image    
        if self.garbageDetectionPipeline:
            self.process_garbage_detection(frame)


        # Run trip hazard detection pipeline on image
        if self.triphazardDetectionPipeline:
            self.process_triphazard_detection(frame)

        if self.spillDetectionPipeline:
            self.process_spill_detection(frame)

        # Run person counting in a zone
        if self.personInZoneCounting:
            self.process_zone_counting()
            

        # delete the values that were tracked
        self.personDetectionPipeline.personBboxes.clear()
        
        # print(json.dumps(self.jsonResultsManager.fullImageResults, indent=4))

        #draw the results on the frame
        drawn_frame = self.drawOnFrames([frame],[self.jsonResultsManager.fullImageResults])
        
        # return drawn frames
        return drawn_frame

    def process_ppe_detection(self, frame, person_bboxes):
        """Run ppe detection on the image

        Args:
            frame (np.array): full Image for ppe detection, but we will crop by person later
            person_bboxes (list of lists): contains information like xmin, ymin,xmax,ymax etc. 
            refer person detection pipeline for better understanding
        """
        # run ppe detection pipeline
        self.ppeDetectionPipeline(
            person_bboxes, frame,)

        # Add ppe results in results json
        self.jsonResultsManager.add_ppe_results(
            self.ppeDetectionPipeline.validatedPpeResults
        )
        self.ppeDetectionPipeline.validatedPpeResults.clear()

    def process_zone_counting(
        self,
    ):
        """
        This will handle checking the number of person in zone
        """
        self.personInZoneCounting.calculate_person_within_zone(self.jsonResultsManager.fullImageResults)

    def process_fire_and_smoke(self, frame):
        """
        Handle fire and smoke detection on frames.

        Args:
            frame (np.array): The current video frame.
        """
        
        self.fireSmokeDetectionPipeline(frame)
        fire_bboxes = self.fireSmokeDetectionPipeline.fire_bboxes
        smoke_bboxes = self.fireSmokeDetectionPipeline.smoke_bboxes
        fire_flag = self.fireSmokeDetectionPipeline.fire_detected
        smoke_flag = self.fireSmokeDetectionPipeline.smoke_detected
        if fire_flag or smoke_flag:
            self.jsonResultsManager.add_fire_smoke_results(fire_bboxes, smoke_bboxes, fire_flag,smoke_flag)

    def process_fall_detection(self, frame , person_bboxes):
        """
        Runs fall detection pipeline and adds the results
        """
        self.fallDetectionPipeline(frame, person_bboxes)
        self.jsonResultsManager.add_fall_results(self.fallDetectionPipeline.fall_result)

    def process_garbage_detection(self,frame):
        """
        Handles garbage detection on the frames
        """
        self.garbageDetectionPipeline(frame)
        self.jsonResultsManager.add_garbage_results(self.garbageDetectionPipeline.garbage_results)
        self.garbageDetectionPipeline.garbage_results.clear()
       
    def process_triphazard_detection(self,frame):
        """
        Handles  trip hazard detection on the frames
        """
        self.triphazardDetectionPipeline(frame)
        #If no object is detected 
        if not self.triphazardDetectionPipeline.detection_results:
            return
        
        self.jsonResultsManager.add_triphazard_results(self.triphazardDetectionPipeline.detection_results)
        self.triphazardDetectionPipeline.detection_results.clear()

    def process_spill_detection(self,frame):
        self.spillDetectionPipeline(frame)
        self.jsonResultsManager.add_spill_results(self.spillDetectionPipeline.spill_results)
        self.spillDetectionPipeline.spill_results.clear()

if __name__ == "__main__":
    processor = VideoProcessor("config/config.json")
    processor()

