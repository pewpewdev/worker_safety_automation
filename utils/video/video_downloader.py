import json
import logging
import os

from dotenv import load_dotenv
import boto3

# Load environment variables from the .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3VideoDownloader:
    """This class is used to download the videos from s3 path to local 
    dir. You might be running the code inside a docker container so it will '
    download the videos inside the container
    
    Attributes:
        bucket_name: Name of bucket where video is
        s3_path: Path to videos dir inside bucket_name
        local_dir: Video save path
        s3_client:  boto3 data downloader object 
        
    """
    def __init__(self, global_config):
        self.bucket_name = global_config["videoDownloader"]["s3BucketName"]
        self.s3_path = global_config["videoDownloader"]["s3VideoPath"]
        self.local_dir = global_config["videoDownloader"]["localVideoPath"]
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        #Start video downloader
        self.download_videos()

    def download_videos(self):
        try:
            # Create the local directory if it doesn't exist
            os.makedirs(self.local_dir, exist_ok=True)
            
            # List objects in the specified S3 bucket and path
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.s3_path
            )
            
            # Check if there are any objects in the specified S3 path
            if "Contents" not in response:
                logger.info(f"No objects found in S3 path '{self.s3_path}'")
                return

            # Iterate over the list of objects in the S3 path
            for obj in response["Contents"]:
                video_key = obj["Key"]
                local_path = os.path.join(self.local_dir, os.path.basename(video_key))
                
                # Skip download if the video already exists locally
                if os.path.exists(local_path):
                    logger.info(f"Video '{video_key}' already exists locally.")
                    continue

                # Download the video from S3 to the local directory
                logger.info(f"Downloading video '{video_key}'...")
                self.s3_client.download_file(self.bucket_name, video_key, local_path)
                logger.info(f"Video '{video_key}' downloaded successfully.")
        
        except Exception as e:
            # Log an error message if the download fails
            logger.error(f"Failed to download videos from S3: {e}")



