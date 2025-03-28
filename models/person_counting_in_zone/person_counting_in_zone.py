import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
from shapely.geometry import Point, Polygon


class personCountInZone:
    """This class will handle all the information related to zones.

    Attributes:
        zonesList: Zone information from the config file
        personInZoneResults: dict to store if the person inside the zone
        validZones: Flag to see if atleast one zone have zone points
    """

    def __init__(self, camera_config):
        self.validZones = False
        # get all the zones
        self.zonesList = self.get_zone_points(camera_config)

    def get_zone_points(self, camera_config):
        """Extracts the zone points from camera config file

        Args:
            camera_config (dict): camera config
        """
        zones_list = list()
        # iterate through all zones in config file
        for each_zone in camera_config["zones"]:
            zones_list.append(
                {
                    "name": camera_config["zones"][each_zone]["name"],
                    "id": camera_config["zones"][each_zone]["id"],
                    "zonePoints": camera_config["zones"][each_zone]["zonePoints"],
                }
            )
            # Throw warning if no zone points were found for this zone
            if not camera_config["zones"][each_zone]["zonePoints"]:
                logging.warning(
                    "no zone points found in zone {}".format(
                        camera_config["zones"][each_zone]["name"]
                    )
                )
            elif camera_config["zones"][each_zone]["zonePoints"]:
                self.validZones = True

        return zones_list
    
    def calculate_person_within_zone(self,fullImageResults):
        """Checks if the detected person is present inside the zone using shapely's
            point within polygon method for ease.

        Args:
            zone_info_list (list of list): Information about all zones defined in camera config
        """
        # Iterate through all the person results
        for index,each_person_dict in enumerate(fullImageResults["personResults"]):
            # Get the center points of the person
            person_center = (
                each_person_dict["centroid"]["x"],
                each_person_dict["centroid"]["y"],
            )
            # Use shaply Point function
            person_center = Point(person_center)
            # Iterate through each zone and see if person is indide the polygon
            for each_zone in self.zonesList:
                if not each_zone["zonePoints"]:
                    continue
                zone_polygon = Polygon(each_zone["zonePoints"])
                # Update results if person is found inside the zone
                if person_center.within(zone_polygon):
                    fullImageResults["personResults"][index]["zoneInformation"]["withinZone"] = True
                    fullImageResults["personResults"][index]["zoneInformation"]["zoneName"] = each_zone["name"]
                    fullImageResults["personResults"][index]["zoneInformation"]["zoneID"] = each_zone["id"]
                    fullImageResults["personCountInZone"][each_zone["name"]] += 1
            
