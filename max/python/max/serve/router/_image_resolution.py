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

import asyncio
import base64
import io
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiofiles
from httpx import (
    AsyncClient,
    HTTPStatusError,
    Timeout,
    TimeoutException,
    TransportError,
)
from max.pipelines.context.exceptions import InputError
from max.serve.config import Settings
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl

logger = logging.getLogger("max.serve")

# ``data:`` payloads at or below this size (in bytes of the base64 string) are
# decoded inline; larger ones are offloaded to a worker thread. Base64 decoding
# is pure-Python CPU work that blocks the event loop, so a multi-MB payload
# decoded inline stalls every other in-flight request (the TTFT culprit in
# CENG-640). The threshold keeps the common small-thumbnail case off the thread
# pool while pushing the expensive large-payload case off the loop.
_DATA_URI_OFFLOAD_THRESHOLD = 256 * 1024

# Read remote media in bounded chunks so a streamed download can be aborted the
# moment it crosses the size cap, instead of buffering the whole (potentially
# huge) body before checking its length.
_HTTP_CHUNK_SIZE = 256 * 1024

# Explicit fetch timeouts. httpx's default is a 5s timeout on *every* operation,
# which is far too aggressive for media downloads: a large video (or a slow
# CDN) routinely needs more than 5s, and any stall longer than that raises a
# ``ReadTimeout`` that — left unhandled — surfaces as an opaque HTTP 500 (and
# the client then retries the whole generation). The read timeout below is
# per-chunk (each ``_HTTP_CHUNK_SIZE`` read must complete within it), not a cap
# on total download time; total size is already bounded by the byte cap. Pick
# values generous enough for slow-but-steady transfers while still failing a
# truly stalled connection in bounded time.
_FETCH_TIMEOUT = Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

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


def _raise_media_too_large(media_kind: str, max_bytes: int) -> None:
    """Raise an :class:`InputError` for media that exceeds its byte cap.

    Mirrors the message format used by :func:`decode_and_validate_images` so the
    early (pre-decode) rejection is indistinguishable from the late one.

    Args:
        media_kind: ``"image"`` or ``"video"`` (used in the error message).
        max_bytes: The cap that was exceeded, in bytes.

    Raises:
        InputError: Always.
    """
    raise InputError(
        f"{media_kind} exceeds the maximum allowed size of "
        f"{max_bytes // (1024 * 1024)}MB"
    )


def _clean_data_uri_base64(data_uri: str) -> str:
    """Extract and normalize the base64 payload of a ``data:`` URI.

    Tolerates the two ways real clients (and the OpenRouter image relay)
    routinely deviate from canonical base64: stripped ``=`` padding and the
    URL-safe alphabet (``-``/``_``). Splitting this off from the decode lets the
    caller estimate the decoded size (and reject oversized payloads) before
    paying for the decode.
    """
    parts = data_uri.split(",", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("data URI has no base64 payload")
    # Some clients wrap long payloads across lines; strip any whitespace.
    b64 = "".join(parts[1].split())
    # Re-add stripped padding (base64 length must be a multiple of 4).
    b64 += "=" * (-len(b64) % 4)
    return b64


def _base64_decoded_size(b64: str) -> int:
    """Return the exact decoded byte length of a padded base64 string.

    Cheap (``O(1)`` arithmetic on the length) so an oversized ``data:`` payload
    can be rejected before allocating and decoding the (potentially huge) bytes.
    """
    if not b64:
        return 0
    padding = b64[-2:].count("=")
    return (len(b64) // 4) * 3 - padding


def _decode_base64(b64: str) -> bytes:
    decoder = (
        base64.urlsafe_b64decode
        if ("-" in b64 or "_" in b64)
        else base64.b64decode
    )
    return decoder(b64)


def _decode_data_uri_base64(data_uri: str) -> bytes:
    """Decode the base64 payload of a ``data:`` image URI."""
    return _decode_base64(_clean_data_uri_base64(data_uri))


def _resolve_data_uri(
    data_uri: str, max_bytes: int | None, media_kind: str
) -> bytes:
    """Validate-then-decode a ``data:`` payload (run on a worker thread).

    Estimates the decoded size from the base64 length and rejects an oversized
    payload *before* decoding it, so a too-large request never allocates the
    full decoded buffer.
    """
    b64 = _clean_data_uri_base64(data_uri)
    if max_bytes is not None and _base64_decoded_size(b64) > max_bytes:
        _raise_media_too_large(media_kind, max_bytes)
    return _decode_base64(b64)


def _settings_int(value: object) -> int | None:
    """Return a positive int settings value, or ``None``.

    Tolerates ``settings`` being ``None`` or a test mock whose attribute is not
    a real ``int`` (in which case the server-level cap simply does not apply).
    """
    return value if isinstance(value, int) and value > 0 else None


async def resolve_image_from_url(
    image_ref: AnyUrl,
    settings: Settings,
    max_bytes: int | None = None,
    media_kind: str | None = None,
) -> bytes:
    """Resolve a media reference into raw bytes, enforcing a byte cap early.

    The effective cap is the smaller of the per-call ``max_bytes`` (e.g. a
    per-model cap from the tokenizer) and the server-level
    :attr:`Settings.max_bytes`; ``0``/``None`` on either side means "no cap from
    that source". When a cap applies, an oversized payload is rejected before
    the bytes are fully materialized: an ``http(s)`` download is aborted as soon
    as the advertised ``Content-Length`` (or the streamed total) crosses the
    cap, and a ``data:`` payload is rejected from its base64 length before it is
    decoded.

    ``media_kind`` (``"image"``/``"video"``) only selects the error wording; it
    falls back to :attr:`Settings.media_kind` and then ``"image"``.
    """
    # Combine the per-call cap with the server-level cap (smaller wins).
    settings_cap = (
        _settings_int(getattr(settings, "max_bytes", None))
        if settings is not None
        else None
    )
    caps = [c for c in (max_bytes, settings_cap) if c is not None and c > 0]
    max_bytes = min(caps) if caps else None
    if media_kind is None:
        settings_kind = getattr(settings, "media_kind", None)
        media_kind = (
            settings_kind if isinstance(settings_kind, str) else "image"
        )

    if image_ref.scheme == "http" or image_ref.scheme == "https":
        # TODO: Evaluate creating a single AsyncClient for the app.
        async with AsyncClient(
            headers=_FETCH_HEADERS, timeout=_FETCH_TIMEOUT
        ) as client:
            try:
                async with client.stream(
                    "GET", str(image_ref), follow_redirects=True
                ) as response:
                    response.raise_for_status()
                    # Fast path: reject up front when the server advertises an
                    # over-cap size, so we never start streaming the body.
                    if max_bytes is not None:
                        advertised = response.headers.get("content-length")
                        if (
                            advertised is not None
                            and advertised.isdigit()
                            and int(advertised) > max_bytes
                        ):
                            _raise_media_too_large(media_kind, max_bytes)
                    # Stream in bounded chunks, aborting the moment the running
                    # total exceeds the cap (covers a missing or lying
                    # Content-Length).
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes(_HTTP_CHUNK_SIZE):
                        total += len(chunk)
                        if max_bytes is not None and total > max_bytes:
                            _raise_media_too_large(media_kind, max_bytes)
                        chunks.append(chunk)
                    images_bytes = b"".join(chunks)
            except HTTPStatusError as e:
                raise ValueError(
                    f"Failed to fetch image: HTTP {e.response.status_code}"
                ) from None
            except TimeoutException:
                # A slow/stalled download must not surface as an opaque 500
                # (which the client then retries). Turn it into a clean input
                # error attributable to the unreachable/slow media source.
                raise InputError(
                    f"timed out fetching {media_kind} from its URL; the source "
                    "may be too slow or the file too large"
                ) from None
            except TransportError as e:
                # Connection reset / DNS / network failure mid-fetch: same
                # treatment as a timeout, a clean input error rather than a 500.
                raise InputError(
                    f"failed to fetch {media_kind} from its URL "
                    f"({type(e).__name__})"
                ) from None
            logger.debug(
                "ResolvedImageUrl: %s -> %d bytes", image_ref, len(images_bytes)
            )
            return images_bytes
    elif image_ref.scheme == "data":
        data_uri = image_ref.unicode_string()
        # Decode off the event loop for large payloads: base64 decoding is
        # CPU-bound pure-Python work that otherwise stalls every concurrent
        # request (CENG-640). Small thumbnails decode inline to skip the
        # thread-pool hop. The size check happens before the decode either way.
        if len(data_uri) > _DATA_URI_OFFLOAD_THRESHOLD:
            images_bytes = await asyncio.to_thread(
                _resolve_data_uri, data_uri, max_bytes, media_kind
            )
        else:
            images_bytes = _resolve_data_uri(data_uri, max_bytes, media_kind)
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
