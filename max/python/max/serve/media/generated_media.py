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
"""Temporary media storage for file-backed generated outputs."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import aiofiles
import numpy as np
from max.interfaces.request.open_responses import (
    OutputImageContent,
    OutputVideoContent,
)

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class StoredMediaAsset:
    """Metadata for a generated media artifact stored on local disk.

    Attributes:
        asset_id: Stable identifier used in MAX's generated-media download
            routes.
        kind: Media category, currently ``"image"`` or ``"video"``.
        path: Absolute path to the saved file on local disk.
        media_type: MIME type returned when serving the file.
        filename: Basename exposed in the HTTP download response.
        size_bytes: File size in bytes used for cache accounting.
        created_at: Unix timestamp for when the asset metadata was created.
        last_accessed_at: Unix timestamp of the most recent successful lookup.
    """

    asset_id: str
    kind: str
    path: Path
    media_type: str
    filename: str
    size_bytes: int
    created_at: float
    last_accessed_at: float


class GeneratedMediaStorageLimitExceeded(RuntimeError):
    """Raised when a generated asset cannot fit into the configured cache."""


class GeneratedMediaStore:
    """Stores generated media files for later download via HTTP."""

    def __init__(
        self,
        root_dir: Path,
        *,
        max_storage_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Initialize per-process media directories and cache accounting."""
        if max_storage_bytes <= 0:
            raise ValueError("max_storage_bytes must be greater than zero.")

        self._images_dir = root_dir / "images"
        self._videos_dir = root_dir / "videos"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir.mkdir(parents=True, exist_ok=True)
        self._images: dict[str, StoredMediaAsset] = {}
        self._videos: dict[str, StoredMediaAsset] = {}
        self._max_storage_bytes = max_storage_bytes
        self._total_size_bytes = 0

    def get_image(self, image_id: str) -> StoredMediaAsset | None:
        """Return a stored image asset by id if it still exists on disk."""
        return self._get_asset(self._images, image_id)

    def get_video(self, video_id: str) -> StoredMediaAsset | None:
        """Return a stored video asset by id if it still exists on disk."""
        return self._get_asset(self._videos, video_id)

    async def save_image_content(
        self, content: OutputImageContent
    ) -> StoredMediaAsset:
        """Persist inline image content to disk and register it in the cache."""
        image_bytes = _decode_output_image_bytes(content)
        image_format = (content.format or "png").lower()
        return await self._save_payload(
            kind="image",
            directory=self._images_dir,
            extension=image_format,
            default_media_type=f"image/{image_format}",
            payload=image_bytes,
        )

    async def save_video_content(
        self,
        content: OutputVideoContent,
        frames_per_second: int,
    ) -> StoredMediaAsset:
        """Encode raw video frames to MP4 bytes and persist the result."""
        video_bytes = self.encode_video_content(content, frames_per_second)
        return await self._save_payload(
            kind="video",
            directory=self._videos_dir,
            extension="mp4",
            default_media_type="video/mp4",
            payload=video_bytes,
        )

    def encode_video_content(
        self,
        content: OutputVideoContent,
        frames_per_second: int,
    ) -> bytes:
        """Encode raw in-memory frames into an MP4 byte payload."""
        if content.frames is None:
            raise ValueError("Cannot encode video content without raw frames.")

        frames = content.frames
        if frames.shape[0] == 0:
            raise ValueError("Cannot encode a video without any frames.")

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            dir=self._videos_dir,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _encode_mp4(
                [frames[t] for t in range(frames.shape[0])],
                tmp_path,
                frames_per_second,
            )
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _save_payload(
        self,
        *,
        kind: str,
        directory: Path,
        extension: str,
        default_media_type: str,
        payload: bytes,
    ) -> StoredMediaAsset:
        """Evict if needed, write a payload to disk, and track the saved asset.

        Workflow:
            1. Validate that the incoming payload can fit in the cache.
            2. Recompute current usage and evict oldest assets until enough
               space is available.
            3. Build metadata for the new file and write it to disk.
            4. Register the saved asset and update cache size accounting.
        """
        size_bytes = len(payload)
        if size_bytes > self._max_storage_bytes:
            raise GeneratedMediaStorageLimitExceeded(
                "Generated media asset exceeds the configured local media "
                f"cache size ({size_bytes} bytes > "
                f"{self._max_storage_bytes} bytes)."
            )

        self._recompute_total_size()
        self._evict_until_fit(size_bytes)

        asset = self._new_asset(
            kind=kind,
            directory=directory,
            extension=extension,
            default_media_type=default_media_type,
            size_bytes=size_bytes,
        )
        try:
            await self._write_file(asset.path, payload)
        except Exception:
            asset.path.unlink(missing_ok=True)
            raise

        target_store = self._images if kind == "image" else self._videos
        target_store[asset.asset_id] = asset
        self._total_size_bytes += asset.size_bytes
        return asset

    def _get_asset(
        self,
        store: dict[str, StoredMediaAsset],
        asset_id: str,
    ) -> StoredMediaAsset | None:
        """Load tracked metadata for an asset and refresh its access time."""
        asset = store.get(asset_id)
        if asset is None:
            return None
        if not asset.path.exists():
            self._remove_asset(store, asset_id, unlink=False)
            return None

        asset = replace(asset, last_accessed_at=time.time())
        store[asset_id] = asset
        return asset

    async def _write_file(self, path: Path, payload: bytes) -> None:
        """Write bytes to disk asynchronously and fsync before returning."""
        async with aiofiles.open(path, "wb") as f:
            await f.write(payload)
            await f.flush()
            await asyncio.to_thread(os.fsync, f.fileno())

    def _new_asset(
        self,
        *,
        kind: str,
        directory: Path,
        extension: str,
        default_media_type: str,
        size_bytes: int,
    ) -> StoredMediaAsset:
        """Create metadata for a future on-disk asset with a fresh id."""
        asset_id = uuid4().hex
        output_path = directory / f"{asset_id}.{extension}"
        media_type = (
            mimetypes.guess_type(output_path.name)[0] or default_media_type
        )
        now = time.time()
        return StoredMediaAsset(
            asset_id=asset_id,
            kind=kind,
            path=output_path,
            media_type=media_type,
            filename=output_path.name,
            size_bytes=size_bytes,
            created_at=now,
            last_accessed_at=now,
        )

    def _evict_until_fit(self, incoming_size_bytes: int) -> None:
        """Remove oldest assets until a new payload can fit in the cache."""
        while self._total_size_bytes + incoming_size_bytes > (
            self._max_storage_bytes
        ):
            candidate = self._oldest_asset()
            if candidate is None:
                raise GeneratedMediaStorageLimitExceeded(
                    "Generated media cache could not free enough space for "
                    "the new asset."
                )

            store, asset_id, asset = candidate
            self._remove_asset(store, asset_id, unlink=True)
            logger.info(
                "Evicted generated %s %s (%d bytes) to free media cache space.",
                asset.kind,
                asset.asset_id,
                asset.size_bytes,
            )

    def _oldest_asset(
        self,
    ) -> tuple[dict[str, StoredMediaAsset], str, StoredMediaAsset] | None:
        """Return the oldest tracked asset across image and video stores."""
        candidates = [
            (store, asset_id, asset)
            for store in (self._images, self._videos)
            for asset_id, asset in store.items()
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: (
                item[2].last_accessed_at,
                item[2].created_at,
                item[2].asset_id,
            ),
        )

    def _remove_asset(
        self,
        store: dict[str, StoredMediaAsset],
        asset_id: str,
        *,
        unlink: bool,
    ) -> None:
        """Drop an asset from tracking and optionally delete its file."""
        asset = store.pop(asset_id, None)
        if asset is None:
            return

        self._total_size_bytes = max(
            0, self._total_size_bytes - asset.size_bytes
        )
        if unlink:
            asset.path.unlink(missing_ok=True)

    def _recompute_total_size(self) -> None:
        """Recalculate aggregate cache usage from tracked asset metadata."""
        self._total_size_bytes = sum(
            asset.size_bytes
            for store in (self._images, self._videos)
            for asset in store.values()
        )


def _decode_output_image_bytes(content: OutputImageContent) -> bytes:
    """Decode inline base64 image content into raw file bytes."""
    if content.image_data is None:
        raise ValueError(
            "Only inline output_image payloads can be persisted to disk."
        )
    return base64.b64decode(content.image_data)


def encode_video_bytes_b64(video_bytes: bytes) -> str:
    """Encode an MP4 byte payload as a base64 response string."""
    return base64.b64encode(video_bytes).decode("utf-8")


def _encode_mp4(
    frames: list[np.ndarray], output_path: Path, frames_per_second: int
) -> None:
    """Encode RGB frames to an MP4 file on disk with PyAV."""
    import av
    import av.video

    height, width = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    stream: av.video.VideoStream = container.add_stream(
        "libx264",
        rate=frames_per_second,
    )
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.options = {"crf": "18", "preset": "medium"}

    try:
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(
                frame_array.astype(np.uint8, copy=False),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
