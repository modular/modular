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
"""Resolve and validate image references from chat-completion requests.

Turns the ``image_url`` / ``video_url`` references in an OpenAI request into raw
image bytes (from ``http(s):``, ``data:``, or ``file:`` URIs) and fully decodes
them once for validation. Kept in its own module so the image-resolution
concern stays focused and out of the much larger ``openai_routes`` handler.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiofiles
from httpx import AsyncClient, HTTPStatusError
from max.pipelines.context.exceptions import InputError
from max.serve.config import Settings
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl

logger = logging.getLogger("max.serve")

# Some media hosts (e.g. Wikimedia, Google Cloud Storage) reject requests that
# carry a default library User-Agent (httpx sends ``python-httpx/...``) with an
# HTTP 403, which turned valid user-supplied image/video URLs into fetch
# failures. Present a common browser User-Agent (and a permissive Accept) so
# fetching from such hosts succeeds.
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def decode_and_validate_images(
    images: list[bytes], max_image_bytes: int | None = None
) -> list[Image.Image]:
    # Fully decode each image so empty, non-image, or truncated/streamed
    # content (e.g. animated or content-negotiated WebP) fails here as a clean
    # 400 instead of reaching the model worker and crashing it with an
    # unhandled PIL error or OSError (HTTP 500). ``Image.open`` is lazy -- it
    # only parses the header -- so a header-valid but undecodable image slips
    # through and later blows up in the tokenizer's ``to_rgb(...)`` ->
    # ``.convert("RGB")`` decode. ``image.load()`` forces that same pixel
    # decode now, while we can still turn the failure into a 400.
    #
    # The decoded images are returned and carried on the request
    # (``TextGenerationRequest.decoded_images``) so the tokenizer reuses them
    # instead of decoding the same bytes a second time. We therefore do not
    # close the images here (no ``with`` block): ``load()`` has already pulled
    # the pixels into memory and the caller owns the decoded image.
    decoded: list[Image.Image] = []
    for image_bytes in images:
        # Optional model-specific cap on resolved bytes (e.g. 10MB).
        if max_image_bytes is not None and len(image_bytes) > max_image_bytes:
            raise InputError(
                "image exceeds the maximum allowed size of "
                f"{max_image_bytes // (1024 * 1024)}MB"
            )
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as e:
            raise InputError("invalid or unreadable image content") from e
        decoded.append(image)
    return decoded


def _decode_data_uri_base64(data_uri: str) -> bytes:
    """Decode the base64 payload of a ``data:`` image URI.

    Tolerates the two ways real clients (and the OpenRouter image relay)
    routinely deviate from canonical base64: stripped ``=`` padding and the
    URL-safe alphabet (``-``/``_``). ``base64.decodebytes`` accepts neither --
    it raises ``binascii.Error`` on missing padding and silently mis-decodes
    URL-safe input to the wrong bytes -- which turned valid image requests into
    400s.
    """
    parts = data_uri.split(",", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("data URI has no base64 payload")
    # Some clients wrap long payloads across lines; strip any whitespace.
    b64 = "".join(parts[1].split())
    # Re-add stripped padding (base64 length must be a multiple of 4).
    b64 += "=" * (-len(b64) % 4)
    decoder = (
        base64.urlsafe_b64decode
        if ("-" in b64 or "_" in b64)
        else base64.b64decode
    )
    return decoder(b64)


async def resolve_image_from_url(
    image_ref: AnyUrl, settings: Settings
) -> bytes:
    if image_ref.scheme == "http" or image_ref.scheme == "https":
        # TODO: Evaluate creating a single AsyncClient for the app.
        async with AsyncClient(headers=_FETCH_HEADERS) as client:
            try:
                response = await client.get(
                    str(image_ref), follow_redirects=True
                )
                response.raise_for_status()
            except HTTPStatusError as e:
                raise ValueError(
                    f"Failed to fetch image: HTTP {e.response.status_code}"
                ) from None
            images_bytes = await response.aread()
            logger.debug(
                "ResolvedImageUrl: %s -> %d bytes", image_ref, len(images_bytes)
            )
            return images_bytes
    elif image_ref.scheme == "data":
        images_bytes = _decode_data_uri_base64(image_ref.unicode_string())
        logger.debug(
            "ResolvedImageB64: %s -> %d bytes",
            str(image_ref)[:16],
            len(images_bytes),
        )
        return images_bytes
    elif image_ref.scheme == "file":
        if settings is None:
            raise ValueError("Settings required for file URI resolution")

        # Parse the file URI.
        parsed = urlparse(str(image_ref))

        # Check host - only allow empty or localhost.
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"File URI with remote host '{parsed.netloc}' is not supported"
            )

        # Extract and decode the path.
        file_path = Path(unquote(parsed.path))

        # Validate against allowed roots.
        allowed_roots = [Path(root) for root in settings.allowed_image_roots]
        if not allowed_roots:
            raise ValueError(
                "File URI access denied: no allowed roots configured"
            )

        # Resolve the path, following symlinks.
        try:
            resolved_path = file_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"File not found: {file_path}") from e

        # Check if it's a directory.
        if resolved_path.is_dir():
            raise ValueError(f"Path is a directory: {resolved_path}")

        # Check if path is within allowed roots.
        path_allowed = False
        for root in allowed_roots:
            try:
                resolved_path.relative_to(root)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            raise ValueError(
                f"Path forbidden: {resolved_path} is outside allowed roots"
            )

        # Read the file with size limit.
        max_bytes = settings.max_local_image_bytes

        async with aiofiles.open(resolved_path, "rb") as f:
            images_bytes = await f.read(max_bytes + 1)
            if len(images_bytes) > max_bytes:
                raise ValueError(
                    f"File exceeds size limit of {max_bytes} bytes"
                )
        logger.debug(
            "ResolvedFileUri: %s -> %d bytes", resolved_path, len(images_bytes)
        )
        return images_bytes
    raise ValueError(f"Invalid image ref '{image_ref}'")
