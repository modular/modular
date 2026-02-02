# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


def fetch_bytes_from_s3(s3_path: str) -> bytes:
    """Fetch bytes from an S3 path.

    Args:
        s3_path: S3 URI in format s3://bucket/key

    Returns:
        Raw bytes content from S3 object

    Raises:
        ImportError: If boto3 is not installed. Install with `pip install boto3`
            or use the benchmark extras: `pip install max[benchmark]`
    """
    import io

    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 operations. "
            "Install it with `pip install boto3` or use `pip install max[benchmark]`"
        ) from e

    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    # Parse S3 path: s3://bucket/key
    path_parts = s3_path[5:].split("/", 1)
    if len(path_parts) != 2:
        raise ValueError(f"Invalid S3 path format: {s3_path}")

    bucket, key = path_parts
    s3 = boto3.client("s3")

    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    return buffer.getvalue()
