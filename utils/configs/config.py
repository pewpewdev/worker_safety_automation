import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class jsonConfigParser:
    def __init__(self, config_file_path):
        self.config = self.load_config(config_file_path)

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}
