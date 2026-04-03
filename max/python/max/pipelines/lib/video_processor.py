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

"""Video helpers for diffusion pipelines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import PIL.Image


class VideoProcessor:
    """Simple PIL-video preprocessor inspired by Diffusers' VideoProcessor."""

    def __init__(self, *, mask: bool = False) -> None:
        self._mask = mask

    @staticmethod
    def pad_video_frames(
        frames: list[PIL.Image.Image],
        num_target_frames: int,
    ) -> list[PIL.Image.Image]:
        """Pad frames using a reflect-like strategy along the time axis."""
        if not frames:
            raise ValueError("Expected at least one frame to pad.")
        if len(frames) >= num_target_frames:
            return frames[:num_target_frames]

        idx = 0
        flip = False
        padded_frames: list[PIL.Image.Image] = []
        while len(padded_frames) < num_target_frames:
            padded_frames.append(frames[idx])
            idx = idx - 1 if flip else idx + 1
            if idx == 0 or idx == len(frames) - 1:
                flip = not flip

        return padded_frames

    @staticmethod
    def compute_segment_layout(
        num_frames: int,
        segment_frame_length: int,
        prev_segment_conditioning_frames: int,
    ) -> tuple[int, int, int]:
        """Compute padded frame count and number of segments."""
        effective_segment_length = (
            segment_frame_length - prev_segment_conditioning_frames
        )
        last_segment_frames = (
            num_frames - prev_segment_conditioning_frames
        ) % effective_segment_length
        num_padding_frames = (
            0
            if last_segment_frames == 0
            else effective_segment_length - last_segment_frames
        )
        num_target_frames = num_frames + num_padding_frames
        num_segments = num_target_frames // effective_segment_length
        return effective_segment_length, num_target_frames, num_segments

    def preprocess_video(
        self,
        video: list[PIL.Image.Image],
        *,
        height: int,
        width: int,
        num_target_frames: int | None = None,
        resample: PIL.Image.Resampling | None = None,
    ) -> npt.NDArray[np.float32]:
        """Preprocess a list of PIL frames into normalized numpy tensors."""
        if not video:
            raise ValueError("Expected non-empty video input.")

        if num_target_frames is not None:
            video = self.pad_video_frames(video, num_target_frames)

        if self._mask:
            resize_mode = PIL.Image.Resampling.NEAREST
            frames = [frame.convert("L") for frame in video]
            if any(frame.size != (width, height) for frame in frames):
                frames = [
                    frame.resize((width, height), resize_mode)
                    for frame in frames
                ]
            output = np.stack(
                [np.asarray(frame, dtype=np.float32) for frame in frames],
                axis=0,
            ).astype(np.float32, copy=False)
            if output.max() > 1.0:
                output /= 255.0
            return output

        resize_mode = resample or PIL.Image.Resampling.LANCZOS
        frames = [frame.convert("RGB") for frame in video]
        if any(frame.size != (width, height) for frame in frames):
            frames = [
                frame.resize((width, height), resize_mode) for frame in frames
            ]
        output = np.stack(
            [
                np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)
                for frame in frames
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        output /= 127.5
        output -= 1.0
        return output


def load_video_frames(video_path: str) -> list[PIL.Image.Image]:
    """Load a video into a list of RGB PIL frames using PyAV."""
    import av

    container = av.open(video_path)
    frames: list[PIL.Image.Image] = []
    for frame in container.decode(video=0):
        frames.append(frame.to_image().convert("RGB"))
    container.close()
    return frames


def save_video(
    frames: list[npt.NDArray[np.uint8]],
    output_path: str,
    fps: int,
) -> None:
    """Encode RGB uint8 frames to mp4 using PyAV."""
    import av
    import av.video

    if not frames:
        raise ValueError("Expected at least one frame to save.")

    height, width = frames[0].shape[:2]
    container = av.open(output_path, mode="w")
    stream: av.video.VideoStream = container.add_stream("libx264", rate=fps)  # type: ignore[assignment]
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.options = {"crf": "18", "preset": "medium"}

    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def video_tensor_to_frames(
    video: npt.NDArray[np.generic],
) -> list[npt.NDArray[np.uint8]]:
    """Convert a single video tensor to a list of HWC uint8 frames."""
    if video.ndim != 4:
        raise ValueError(
            "Expected a single video tensor with 4 dimensions, "
            f"got shape {video.shape}."
        )

    if video.shape[0] in (1, 3):
        frames = np.transpose(video, (1, 2, 3, 0))
    elif video.shape[-1] in (1, 3):
        frames = video
    else:
        raise ValueError(
            "Unsupported video tensor layout; expected channels-first "
            f"or channels-last RGB data, got shape {video.shape}."
        )

    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames * 0.5 + 0.5, 0.0, 1.0)
        frames = np.rint(frames * 255.0).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8, copy=False)

    return [frame for frame in frames]
