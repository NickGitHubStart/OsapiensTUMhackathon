"""Download the makeathon dataset from the public S3 bucket."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_s3_folder(
    bucket_name: str,
    folder_name: str,
    local_dir: str = "./data",
    skip_existing: bool = True,
) -> None:
    """Download a folder from an S3 bucket into a local directory."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = folder_name.strip("/")
    if prefix:
        prefix = f"{prefix}/"

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                logger.warning(
                    "No objects found in folder '%s' in bucket '%s'",
                    folder_name,
                    bucket_name,
                )
                return

            for obj in page["Contents"]:
                key = obj["Key"]
                target = local_path / key

                if key.endswith("/") or key == prefix:
                    continue

                if skip_existing and target.exists():
                    logger.info("Skipping existing %s", target)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading %s -> %s", key, target)
                s3.download_file(bucket_name, key, str(target))

        logger.info(
            "Successfully downloaded folder '%s' from bucket '%s' to '%s'",
            folder_name,
            bucket_name,
            local_dir,
        )

    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        raise
    except ClientError as exc:
        logger.error("AWS client error: %s", exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a folder from a public S3 bucket using boto3."
    )
    parser.add_argument(
        "--bucket-name",
        default="osapiens-terra-challenge",
        help="Name of the S3 bucket",
    )
    parser.add_argument(
        "--folder-name",
        default="makeathon-challenge",
        help="Name of the folder inside the S3 bucket",
    )
    parser.add_argument(
        "--local-dir",
        default="./data",
        help="Local directory to save files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist",
    )

    args = parser.parse_args()

    download_s3_folder(
        args.bucket_name,
        args.folder_name,
        args.local_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
