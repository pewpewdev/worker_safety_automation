import os
import pickle
import sqlite3
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("models/reid")
from torchreid.utils import FeatureExtractor



class reID:
    """
    This class implements a Person Re-Identification (ReID) system.
    To implement reid we wait till a new track_id comes into observation. Instead of blindly trusting our tracker,
    we first check if the new person is really a new person or onld person which was either occluded or went outside the
    camera view.
    We check new track_id's person's  crop and we calculate the cosine similarity with the last 'x' feature maps for every person ID.
    if we find confidence of match is greater than  threshold, this is not a valid new person
    The feature map that has the highest cosine similarity with the current crop will be considered as the match.
    The previously extracted feature maps will be stored in a local database, which will be continually updated with newly
    extracted feature maps from person crops."

    This class implements functionalities related to person re-identification,
    including feature extraction, database management, and re-identification process.

    Attributes:
        device = cpu or cuda
        feature_extractor: Feature extraction model used for extracting image features.
        local_database_name (str): Name of the local database used for storing feature maps.
        number_of_features_for_reid (int): Number of features used for re-identification.
        device(str) device at which we want to perform reid
        feature_extractor(FeatureExtractor object): run images through them to extract features
        tracklets: track history of every person
        conf_threshold(float) = Confidence threshold for a valid old match

    Methods:
        init_feature_extractor():  method to init feature extraction model
        init_database(): Placeholder method for initializing the local database.
        get_feature_maps_from_feature_extractor(): Placeholder method to get feature maps from the extractor.
        add_feature_maps_to_database(): Placeholder method to add feature maps to the database.
        get_feature_maps_from_database(): Placeholder method to retrieve feature maps from the database.
        vstack_feature_maps_from_database: stack into tensor for faster calculation leveraging pytorch tensor
        calculate_cosine_similarity: calculate cosine similarity between

    Order of execution:
        1. __call__
        2. get_feature_maps_from_feature_extractor
        3. get_feature_maps_from_database
        4. vstack_feature_maps_from_database
        5. add_feature_maps_to_database
        6. calculate_cosine_similarity


    """

    def __init__(self, camera_config, main_config):
        """
        Initialize a ReID (Person Re-Identification) system.

        Args:
            camera_config (dict): Configuration parameters for the camera.
            main_config (dict): Main configuration file

        """
        self.device = main_config["reIdModel"]["device"]

        self.local_database_name = os.path.splitext(camera_config["camID"])[0] + ".db"
        self.init_database()
        self.number_of_features_for_reid = main_config["reIdModel"]["noOfFrameFeatures"]
        self.feature_extractor = self.init_feature_extractor(
            main_config["reIdModel"]["modelType"],
            os.path.join(
                main_config["modelsDir"], main_config["reIdModel"]["modelName"]
            ),
        )
        self.tracklets = dict()
        self.conf_threshold = main_config["reIdModel"]["confidence"]

    def init_feature_extractor(self, model_type, model_path):
        """
        Initialize the feature extractor for extraction person crops feature
        Args:
            model_type (str): Name of model we want to use for feature extractor
            model_path (str): Path of model weight path

        """

        return FeatureExtractor(
            model_name=model_type, model_path=model_path, device=self.device
        )

    def init_database(
        self,
    ):
        """
        This function initializes the a local db to store the feature maps
        given by self.feature_extractor
        """
        conn = sqlite3.connect(self.local_database_name)
        c = conn.cursor()

        # Create a table for storing feature maps
        c.execute(
            """CREATE TABLE IF NOT EXISTS FeatureMaps
                    (id INTEGER PRIMARY KEY,
                    primary_key TEXT,
                    feature_map BLOB)"""
        )

        conn.commit()
        conn.close()

    def get_feature_maps_from_feature_extractor(self, img_crop_list):
        """
        This function will take one image crop in the form of list and use
        self.feature_extractor to extract the image features. It returns the extracted
        features in pytorch tensor

        Args:
            img_crop_list : list of single image
        Returns:
            torch.tensor: feature map in the form of tensor

        """
        return self.feature_extractor(img_crop_list)

    def add_feature_maps_to_database(self, feature_map, primary_key):
        """
        Adds the feature map into our database
        """
        conn = sqlite3.connect(self.local_database_name)
        c = conn.cursor()

        # Serialize torch tensor to bytes using pickle
        feature_map_bytes = pickle.dumps(feature_map)

        # Insert feature map into the database
        c.execute(
            """INSERT INTO FeatureMaps (primary_key, feature_map)
                VALUES (?, ?)""",
            (int(primary_key), feature_map_bytes),
        )

        conn.commit()
        conn.close()

    def get_feature_maps_from_database(self):
        """
        Retrive last x features from db for doing cosine similarity

        """
        conn = sqlite3.connect(self.local_database_name)
        c = conn.cursor()

        # Retrieve last 10 feature maps for each primary key using a subquery
        c.execute(
            """SELECT id, primary_key, feature_map
                    FROM (
                        SELECT id, primary_key, feature_map,
                                ROW_NUMBER() OVER (PARTITION BY primary_key ORDER BY id DESC) AS row_num
                        FROM FeatureMaps
                    ) AS numbered
                    WHERE row_num <= ?""",
            (self.number_of_features_for_reid,),
        )

        rows = c.fetchall()
        result = {}

        for row in rows:
            primary_key = row[1]
            feature_map_bytes = row[2]

            # Deserialize feature map bytes back to torch tensor
            feature_map = pickle.loads(feature_map_bytes)

            if primary_key not in result:
                result[primary_key] = []

            result[primary_key].append(feature_map)

        conn.close()

        return result

    def vstack_feature_maps_from_database(self, feature_maps):
        """
        Perform vstacking of of feature maps from db for faster calculation.
        Idea here is we query feature maps from database in form of dict and vstack
        all the values for every key.
        sample featuremaps = { key1: [t1,t2,t3],
                                key2: [t1,t2,t3],}
        output:[[t1,t2,t3],
                [t1,t2,t3]]
        """
        tensors_by_key = [torch.stack(feature_maps[key]) for key in feature_maps]
        output = torch.stack(tensors_by_key, dim=1)
        return output.permute(1, 0, 2, 3)

    def calculate_cosine_similarity(self, new_tensor, old_tensor):
        """This will calaculate the cosine similarity with last x
            feature maps and current image feature maps. If you have value 10 for
            main_config["reIdModel"]["noOfFrameFeatures"] that means we take last 10 feature maps
            from every person and calculate cosine similarity with each of 10 feature maps.
            After computing each similarity score, we will average them

        Args:
            new_tensor (torch.tensor): New person feature map
            old_tensor (torch.tensor): last x feature maps from every person

        Returns:
            max index (int): Maximum index value
            average_similarity (float): maximim similarity score
        """
        # Reshape tensor 'new_tensor' to match the dimensions of 'old_tensor'
        new_tensor = new_tensor.unsqueeze(0).unsqueeze(2)

        # Compute cosine similarity between 'new_tensor' and 'old_tensor' along the last dimension (512)
        cos_sim = torch.cosine_similarity(new_tensor, old_tensor, dim=-1)

        # Squeeze the output tensor to remove the singleton dimension
        cos_sim = cos_sim.squeeze(-1)
        # get avg of similarities
        average_similarity = cos_sim.mean(dim=1)
        # Max index of the highest similarity score
        max_index = torch.argmax(average_similarity, dim=0)
        # get keys of tracklets
        keys_of_tracklets = list(self.tracklets.keys())
        # return index and value
        return keys_of_tracklets[max_index.item()], average_similarity[max_index]

    def perform_reid(self, cropped_person, person_box, track_id):
        """Perform reid when a new track id is detected for the person


        Args:
            cropped_person (np.array): person crop for which new track id is discovered
            person_box (list): person co-ord bounding box

        Returns:
            track_id (int): updated track id or orignal track id
        """
        # get feature map for current person crop and feature maps from db
        images_list_feature_maps = self.get_feature_maps_from_feature_extractor(
            cropped_person
        )
        old_feature_maps_from_db = self.get_feature_maps_from_database()
        # Sometime you will have very less number of feature (eg: first frame will not have any featured)
        # so using try and except statement
        try:
            stacked_feature_maps = self.vstack_feature_maps_from_database(
                old_feature_maps_from_db
            )
        except:
            self.tracklets[track_id] = []
            self.tracklets[track_id].append(
                (person_box[0], person_box[1], person_box[2], person_box[3])
            )
            return track_id
        # Calculate cosine similarity with new person crop and old person feature maps
        updated_track_id, confidence = self.calculate_cosine_similarity(
            images_list_feature_maps, stacked_feature_maps
        )
        # if confidennce is greater than threshold, assign old track id predicted by reid model
        if confidence >= self.conf_threshold:
            track_id = updated_track_id
            self.tracklets[track_id].append(
                (person_box[0], person_box[1], person_box[2], person_box[3])
            )
            return track_id
        # since confidence is less than predefined threshold, we assume this person is indeed a new person
        else:
            self.tracklets[track_id] = []
            self.tracklets[track_id].append(
                (person_box[0], person_box[1], person_box[2], person_box[3])
            )
        return track_id

    def __call__(self, person_boxes, image):
        """
        Perform person re-identification (ReID) given input images.

        Args:
            person_boxes(list of list): person detection results
            images (np.array): full image .
        """
        # iterate through all the person box
        for index, person_box in enumerate(person_boxes):
            # get track ids
            track_id = person_box[4]
            # crop the person image from big image
            cropped_person = image[
                person_box[1] : person_box[3], person_box[0] : person_box[2]
            ]

            # Perform reid if new track id discovered
            if track_id not in self.tracklets:
                updated_track_id = self.perform_reid(
                    cropped_person, person_box, track_id
                )

                # Update the track_id of this person.
                person_boxes[index][4] = updated_track_id
            # since we did not not find any new track_id, we don't check for re-id
            else:
                self.tracklets[track_id].append(
                    (person_box[0], person_box[1], person_box[2], person_box[3])
                )

            # extract the feature maps to be stored in db
            images_list_feature_maps = self.get_feature_maps_from_feature_extractor(
                cropped_person
            )
            # store feature maps in db

            self.add_feature_maps_to_database(
                images_list_feature_maps, person_boxes[index][4]
            )
