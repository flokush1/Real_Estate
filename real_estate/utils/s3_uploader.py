import os
import re
from typing import List

from real_estate.constant import (
    ARTIFACT_DIR,
    APT_MODEL_TRAINER_DIR,
    APT_RENT_MODEL_TRAINER_DIR,
    BF_MODEL_TRAINER_DIR,
    BF_RENT_MODEL_TRAINER_DIR,
    PLOT_MODEL_TRAINER_DIR,
    S3_AUTO_CREATE_BUCKET,
    S3_BUCKET,
    S3_PREFIX,
    S3_REGION,
)
from real_estate.logging.logger import logging

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


def _ensure_bucket(client, bucket_name: str) -> None:
    try:
        client.head_bucket(Bucket=bucket_name)
        logging.info(f"S3 bucket already exists: {bucket_name}")
        return
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

    if not S3_AUTO_CREATE_BUCKET:
        logging.warning("S3 bucket does not exist and auto-create is disabled.")
        return

    if S3_REGION == "us-east-1":
        client.create_bucket(Bucket=bucket_name)
    else:
        client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": S3_REGION},
    )
    logging.info(f"Created S3 bucket: {bucket_name} ({S3_REGION})")


def upload_artifact_pkls(
    artifact_dir: str = ARTIFACT_DIR,
    bucket: str | None = None,
    prefix: str | None = None,
    categories: list[str] | None = None,
    subcategories: list[str] | None = None,
) -> List[str]:
    """Upload latest versioned (vN) .pkl files under artifact_dir to S3.

    Files are grouped under apt/sell, apt/rent, bf/sell, bf/rent, plot prefixes
    based on trainer dirs. Only the highest vN per group is uploaded. Rent
    artifacts can be unversioned and are uploaded without a vN prefix.
    Returns the list of uploaded S3 keys.
    """
    bucket_name = bucket or S3_BUCKET
    if not bucket_name:
        logging.info("S3 bucket not configured; skipping artifact upload.")
        return []

    if boto3 is None:
        logging.warning("boto3 is not installed; skipping S3 upload.")
        return []

    prefix_value = S3_PREFIX if prefix is None else prefix
    prefix_value = prefix_value.strip("/") if prefix_value else ""
    s3_prefix = f"{prefix_value}/" if prefix_value else ""

    category_map = {
        APT_MODEL_TRAINER_DIR: ("apt", "sell"),
        APT_RENT_MODEL_TRAINER_DIR: ("apt", "rent"),
        BF_MODEL_TRAINER_DIR: ("bf", "sell"),
        BF_RENT_MODEL_TRAINER_DIR: ("bf", "rent"),
        PLOT_MODEL_TRAINER_DIR: ("plot", None),
    }

    version_re = re.compile(r"^v(\d+)(?:/|$)")
    latest_by_category: dict[tuple[str, str | None], int] = {}
    files_index: List[tuple[str, tuple[str, str | None], int, str]] = []
    for root, _, filenames in os.walk(artifact_dir):
        for filename in filenames:
            if not filename.lower().endswith(".pkl"):
                continue
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, artifact_dir).replace(os.sep, "/")
            top_dir, _, remainder = rel_path.partition("/")
            if top_dir not in category_map:
                continue

            category, subcategory = category_map[top_dir]
            match = version_re.match(remainder)
            if match:
                version_num = int(match.group(1))
                remainder_for_s3 = remainder
            elif subcategory == "rent":
                version_num = 0
                remainder_for_s3 = remainder
            else:
                continue

            group_key = (category, subcategory)
            latest_by_category[group_key] = max(
                version_num, latest_by_category.get(group_key, -1)
            )
            files_index.append(
                (local_path, group_key, version_num, remainder_for_s3)
            )

    if not files_index:
        logging.info("No versioned .pkl files found under artifact; skipping S3 upload.")
        return []

    files_to_upload: List[tuple[str, str]] = []
    for local_path, group_key, version_num, remainder in files_index:
        if latest_by_category.get(group_key, -1) != version_num:
            continue
        category, subcategory = group_key
        if categories and category not in categories: 
            continue
        if subcategories and subcategory not in subcategories:
            continue
        if subcategory:
            s3_key = f"{s3_prefix}{category}/{subcategory}/{remainder}"
        else:
            s3_key = f"{s3_prefix}{category}/{remainder}"
        files_to_upload.append((local_path, s3_key))

    if not files_to_upload:
        logging.info("No latest-version .pkl files found; skipping S3 upload.")
        return []

    client = boto3.client("s3", region_name=S3_REGION)
    _ensure_bucket(client, bucket_name)
    uploaded_keys: List[str] = []
    for local_path, s3_key in files_to_upload:
        client.upload_file(local_path, bucket_name, s3_key)
        uploaded_keys.append(s3_key)
        logging.info(f"Uploaded {local_path} -> s3://{bucket_name}/{s3_key}")

    logging.info(f"S3 upload complete: {len(uploaded_keys)} file(s)")
    return uploaded_keys
