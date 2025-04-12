import os
import subprocess

from networksecurity.logging import logger


class S3Sync:
    def sync_folder_to_s3(self, local_folder: str, bucket_url: str) -> None:
        """
        Syncs _(uploads)_ a local folder to an S3 bucket using the AWS CLI.
        This method uses the AWS CLI command `aws s3 sync` to perform the sync operation.

        Args:
            local_folder (str): The local folder to sync to S3.
            bucket_url (str): The S3 bucket URL (e.g., s3://bucket-name/path).
        """
        try:
            subprocess.run(
                ["aws", "s3", "sync", local_folder, bucket_url],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Synced folder {local_folder} to S3 bucket {bucket_url}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to sync to S3: {e.stderr}")

    def sync_folder_from_s3(self, bucket_url: str, local_folder: str) -> None:
        """
        Syncs _(downloads)_ an S3 bucket to a local folder using the AWS CLI.
        This method uses the AWS CLI command `aws s3 sync` to perform the sync operation.

        Args:
            bucket_url (str): The S3 bucket URL (e.g., s3://bucket-name/path).
            local_folder (str): The local folder to sync to.
        """
        try:
            subprocess.run(
                ["aws", "s3", "sync", bucket_url, local_folder],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Synced S3 bucket {bucket_url} to local folder {local_folder}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to sync from S3: {e.stderr}")
