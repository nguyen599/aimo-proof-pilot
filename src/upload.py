import argparse
import logging
import mimetypes
import os
import posixpath
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import boto3

S3_BUCKET = "aimo-proof-pilot"
S3_PREFIX = "olmo3-32b-ckpt-1"
DEFAULT_S3_URL = f"s3://{S3_BUCKET}/{S3_PREFIX}"
AWS_REGION = "us-east-1"
AWS_ENDPOINT_URL = None
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN", "")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a result file or directory to S3."
    )
    parser.add_argument(
        "--s3_url",
        default=DEFAULT_S3_URL,
        help=(
            "Destination. Use s3://bucket/prefix for recursive folder uploads, "
            "or an https:// presigned PUT URL for a single archive/file upload."
        ),
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Path to the local file or directory to upload.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_s3_url(s3_url: str) -> tuple[str, str] | None:
    parsed = urllib.parse.urlparse(s3_url)
    if parsed.scheme != "s3":
        return None
    if not parsed.netloc:
        raise ValueError(f"S3 URL is missing bucket name: {s3_url}")
    return parsed.netloc, parsed.path.lstrip("/")


def iter_source_files(source_path: Path) -> list[tuple[Path, str]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if source_path.is_file():
        return [(source_path, source_path.name)]
    if not source_path.is_dir():
        raise ValueError(f"Source path must be a file or directory: {source_path}")

    files: list[tuple[Path, str]] = []
    for path in sorted(source_path.rglob("*")):
        if path.is_file():
            files.append((path, path.relative_to(source_path).as_posix()))
    return files


def s3_join(prefix: str, relative_key: str) -> str:
    prefix = prefix.strip("/")
    relative_key = relative_key.strip("/")
    if not prefix:
        return relative_key
    return posixpath.join(prefix, relative_key)


def build_s3_client():
    if (
        not AWS_ACCESS_KEY_ID
        or not AWS_SECRET_ACCESS_KEY
        or AWS_ACCESS_KEY_ID.startswith("REPLACE_WITH_")
        or AWS_SECRET_ACCESS_KEY.startswith("REPLACE_WITH_")
    ):
        raise RuntimeError(
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables before using s3:// uploads."
        )

    client_kwargs = {
        "region_name": AWS_REGION,
        "endpoint_url": AWS_ENDPOINT_URL,
    }
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        client_kwargs.update(
            {
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
                "aws_session_token": AWS_SESSION_TOKEN or None,
            }
        )
    return boto3.client("s3", **client_kwargs)


def upload_to_s3(source_path: Path, bucket: str, prefix: str) -> None:
    files = iter_source_files(source_path)
    if not files:
        raise ValueError(f"No files found to upload under {source_path}")

    logging.info("Uploading %d file(s) from %s to s3://%s/%s", len(files), source_path, bucket, prefix)
    s3_client = build_s3_client()

    for local_path, relative_key in files:
        target_key = s3_join(prefix, relative_key)
        size_bytes = local_path.stat().st_size
        logging.info("Uploading %s (%d bytes) -> s3://%s/%s", local_path, size_bytes, bucket, target_key)

        content_type, _ = mimetypes.guess_type(local_path.name)
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            s3_client.upload_file(str(local_path), bucket, target_key, ExtraArgs=extra_args)
        else:
            s3_client.upload_file(str(local_path), bucket, target_key)

    logging.info("Upload completed successfully")


def prepare_presigned_upload_path(source_path: Path, temp_dir: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise ValueError(f"Source path must be a file or directory: {source_path}")

    archive_base = temp_dir / source_path.name
    archive_path = shutil.make_archive(
        base_name=str(archive_base),
        format="zip",
        root_dir=source_path.parent,
        base_dir=source_path.name,
    )
    return Path(archive_path)


def upload_to_presigned_url(source_path: Path, s3_url: str) -> None:
    with tempfile.TemporaryDirectory(prefix="submission-upload-") as temp_name:
        upload_path = prepare_presigned_upload_path(source_path, Path(temp_name))
        size_bytes = upload_path.stat().st_size
        logging.info("Uploading %s (%d bytes) to presigned URL", upload_path, size_bytes)
        with upload_path.open("rb") as file_obj:
            request = urllib.request.Request(
                s3_url,
                data=file_obj,
                method="PUT",
                headers={
                    "Content-Length": str(size_bytes),
                    "Content-Type": "application/octet-stream",
                },
            )
            with urllib.request.urlopen(request, timeout=3600) as response:
                status = response.getcode()
                if status < 200 or status >= 300:
                    raise RuntimeError(f"S3 upload failed with HTTP status {status}")
    logging.info("Upload completed successfully")


def main() -> int:
    configure_logging()
    args = parse_args()
    source_path = Path(args.source_dir).expanduser().resolve()

    try:
        parsed_s3 = parse_s3_url(args.s3_url)
        if parsed_s3 is not None:
            bucket, prefix = parsed_s3
            upload_to_s3(source_path, bucket, prefix)
        else:
            upload_to_presigned_url(source_path, args.s3_url)
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        logging.error("Upload failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
