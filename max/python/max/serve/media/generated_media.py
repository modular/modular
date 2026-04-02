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

import base64
import mimetypes
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from max.interfaces.request.open_responses import OutputImageContent
from PIL import Image


@dataclass(frozen=True)
class StoredMediaAsset:
    """Metadata for a generated file persisted on disk."""

    asset_id: str
    path: Path
    media_type: str
    filename: str


class GeneratedMediaStore:
    """Stores generated media files for later download via HTTP."""

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir
        self._images_dir = root_dir / "images"
        self._videos_dir = root_dir / "videos"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir.mkdir(parents=True, exist_ok=True)
        self._images: dict[str, StoredMediaAsset] = {}
        self._videos: dict[str, StoredMediaAsset] = {}

    def get_image(self, image_id: str) -> StoredMediaAsset | None:
        return self._images.get(image_id)

    def get_video(self, video_id: str) -> StoredMediaAsset | None:
        return self._videos.get(video_id)

    def save_image_content(
        self, content: OutputImageContent
    ) -> StoredMediaAsset:
        image_bytes = _decode_output_image_bytes(content)
        image_format = (content.format or "png").lower()
        asset = self._write_asset(
            directory=self._images_dir,
            extension=image_format,
            default_media_type=f"image/{image_format}",
            payload=image_bytes,
        )
        self._images[asset.asset_id] = asset
        return asset

    def save_video_frames(
        self,
        frame_contents: list[OutputImageContent],
        frames_per_second: int,
    ) -> StoredMediaAsset:
        if not frame_contents:
            raise ValueError("Cannot save a video without any frames.")

        video_bytes = self.encode_video_frames(
            frame_contents=frame_contents,
            frames_per_second=frames_per_second,
        )
        output_path = self._videos_dir / f"{uuid4().hex}.mp4"
        output_path.write_bytes(video_bytes)

        asset = StoredMediaAsset(
            asset_id=output_path.stem,
            path=output_path,
            media_type="video/mp4",
            filename=output_path.name,
        )
        self._videos[asset.asset_id] = asset
        return asset

    def encode_video_frames(
        self,
        frame_contents: list[OutputImageContent],
        frames_per_second: int,
    ) -> bytes:
        if not frame_contents:
            raise ValueError("Cannot encode a video without any frames.")

        frames = [
            _decode_output_image_frame(content) for content in frame_contents
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            dir=self._videos_dir,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _encode_mp4(frames, tmp_path, frames_per_second)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _write_asset(
        self,
        directory: Path,
        extension: str,
        default_media_type: str,
        payload: bytes,
    ) -> StoredMediaAsset:
        asset_id = uuid4().hex
        output_path = directory / f"{asset_id}.{extension}"
        output_path.write_bytes(payload)
        media_type = (
            mimetypes.guess_type(output_path.name)[0] or default_media_type
        )
        return StoredMediaAsset(
            asset_id=asset_id,
            path=output_path,
            media_type=media_type,
            filename=output_path.name,
        )


def _decode_output_image_bytes(content: OutputImageContent) -> bytes:
    if content.image_data is None:
        raise ValueError(
            "Only inline output_image payloads can be persisted to disk."
        )
    return base64.b64decode(content.image_data)


def encode_video_bytes_b64(video_bytes: bytes) -> str:
    return base64.b64encode(video_bytes).decode("utf-8")


def _decode_output_image_frame(content: OutputImageContent) -> np.ndarray:
    image_bytes = _decode_output_image_bytes(content)
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _encode_mp4(
    frames: list[np.ndarray], output_path: Path, frames_per_second: int
) -> None:
    import av
    import av.video

    height, width = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    stream: av.video.VideoStream = container.add_stream(  # type: ignore[assignment]
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
