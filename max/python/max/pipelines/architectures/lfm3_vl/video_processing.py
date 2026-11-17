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

"""Qwen3VL video preprocessing in pure numpy/PIL (no torch).

A clip is not a sequence of independent images. Two rules make the video path
structurally different from :mod:`~.tokenizer`'s image path, and both are easy
to lose by reusing it:

- The resize depends on the frame count. ``h_bar``/``w_bar`` are solved against
  a ``t_bar * h_bar * w_bar`` budget, so the same source resolution resizes
  differently at 16 frames than at 32, and a ``(width, height)``-only sizing
  call cannot describe a clip.
- Each patch row spans two *distinct* consecutive frames. The image path fills
  the same 1536-wide row by repeating its single frame across both temporal
  slots; reusing that for video discards all motion, leaving the model to
  answer from frame 0 alone.

Ported from ``transformers`` 5.12.1
``transformers/models/qwen3_vl/video_processing_qwen3_vl.py`` and
``processing_qwen3_vl.py``, parameterized by the checkpoints' own
``video_preprocessor_config.json``.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.pipelines.context import open_video_container
from max.profiler import traced
from PIL import Image

QWEN3VL_VIDEO_MIN_PIXELS = 4096
"""Qwen3VL video ``size.shortest_edge``: the whole-clip pixel floor.

An order of magnitude below the image path's 65536, and applied to
``t_bar * h_bar * w_bar`` rather than to one frame's area.
"""

QWEN3VL_VIDEO_MAX_PIXELS = 25165824
"""Qwen3VL video ``size.longest_edge``: the whole-clip pixel ceiling.

Bounds a clip at ``max_pixels / (temporal_patch_size * patch_size**2 *
merge_size**2)`` = 12288 LLM tokens whatever its frame count. Sizing a clip
with the image ceiling instead produces roughly four times the embeddings a
router spliced placeholders for.
"""

QWEN3VL_VIDEO_FPS = 2.0
"""Frames sampled per second of source video."""

QWEN3VL_VIDEO_MIN_FRAMES = 4
"""Sampling floor, itself capped by the clip's own frame count."""

QWEN3VL_VIDEO_MAX_FRAMES = 768
"""Sampling ceiling."""

QWEN3VL_DEFAULT_CONTAINER_FPS = 24.0
"""Frame rate assumed when the container declares none, as the reference does."""


@traced
def video_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_patch_size: int = 2,
    factor: int = 32,
    min_pixels: int = QWEN3VL_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3VL_VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    """Solves a whole clip's resized frame size against the clip pixel budget.

    Diverges from the image ``smart_resize`` in three ways that are easy to get
    wrong:

    - ``beta`` divides by the raw ``num_frames`` while the budget comparison
      uses the temporally padded ``t_bar``, so the two disagree for an odd
      frame count.
    - A side below one ``factor`` raises; the image function clamps up.
    - There is no post-resize total-pixel rejection.

    Rounding is Python's banker's rounding, as in the reference: ``round(22.5)``
    is 22, which is what lands a 720-pixel side on 704 rather than 736.

    Args:
        num_frames: Sampled frame count, before temporal padding.
        height: Source frame height in pixels.
        width: Source frame width in pixels.
        temporal_patch_size: Frames per temporal patch.
        factor: ``patch_size * merge_size``; both sides land on a multiple.
        min_pixels: Whole-clip pixel floor.
        max_pixels: Whole-clip pixel ceiling.

    Returns:
        The ``(h_bar, w_bar)`` every frame in the clip resizes to.

    Raises:
        ValueError: If a side is below ``factor``, or the aspect ratio exceeds
            200.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than "
            f"factor:{factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_patch_size) * temporal_patch_size

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def sample_frame_indices(
    total_num_frames: int,
    container_fps: float | None,
    *,
    fps: float = QWEN3VL_VIDEO_FPS,
    min_frames: int = QWEN3VL_VIDEO_MIN_FRAMES,
    max_frames: int = QWEN3VL_VIDEO_MAX_FRAMES,
) -> npt.NDArray[np.int64]:
    """Picks the source frame indices to sample from a clip.

    The target count comes from the container's own frame rate rather than the
    clip's length, so a 30 fps and a 60 fps clip of equal duration sample the
    same number of frames.

    ``linspace(...).round()`` is numpy's round-half-to-even, so a midpoint
    index lands on its even neighbour. Rounding half up instead shifts
    individual frames by one and silently changes which frames a clip is
    answered from.

    Args:
        total_num_frames: Frames the container holds.
        container_fps: The container's declared frame rate, or ``None``.
        fps: Frames to sample per second of source video.
        min_frames: Sampling floor, itself capped by ``total_num_frames``.
        max_frames: Sampling ceiling.

    Returns:
        The sampled source frame indices, ascending.

    Raises:
        ValueError: If the clip holds no frames.
    """
    if total_num_frames <= 0:
        raise ValueError("Video contains no decodable frames.")

    effective_fps = (
        container_fps
        if container_fps is not None and container_fps > 0
        else QWEN3VL_DEFAULT_CONTAINER_FPS
    )
    num_frames = int(total_num_frames / effective_fps * fps)
    num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)

    return (
        np.linspace(0, total_num_frames - 1, num_frames)
        .round()
        .astype(np.int64)
    )


def clip_grid(
    num_frames: int,
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = QWEN3VL_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3VL_VIDEO_MAX_PIXELS,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Derives a clip's patch grid and resized frame size.

    ``grid_t`` counts temporal *patches*, not frames -- half the padded frame
    count. The reference's ``replace_video_token`` calls the same quantity
    ``num_frames``, which reads as frames but is ``video_grid_thw[0]``.

    Args:
        num_frames: Sampled frame count, before temporal padding.
        height: Source frame height in pixels.
        width: Source frame width in pixels.
        patch_size: Vision patch size.
        merge_size: Spatial merge factor.
        temporal_patch_size: Frames per temporal patch.
        min_pixels: Whole-clip pixel floor.
        max_pixels: Whole-clip pixel ceiling.

    Returns:
        A ``((grid_t, grid_h, grid_w), (h_bar, w_bar))`` pair.
    """
    h_bar, w_bar = video_smart_resize(
        num_frames,
        height,
        width,
        temporal_patch_size=temporal_patch_size,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    t_bar = math.ceil(num_frames / temporal_patch_size) * temporal_patch_size
    return (
        (
            t_bar // temporal_patch_size,
            h_bar // patch_size,
            w_bar // patch_size,
        ),
        (h_bar, w_bar),
    )


@traced
def patchify_clip(
    frames: Sequence[Image.Image],
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = QWEN3VL_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3VL_VIDEO_MAX_PIXELS,
) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
    """Resizes, normalizes and patchifies one clip's sampled frames.

    Rows run ``grid_t`` outermost, then merge blocks, then the within-block
    ``(mh, mw)`` -- the flatten order of the reference's
    ``permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)``. Within a row the layout is
    ``[channel][temporal][py][px]``, and the two temporal slots hold two
    distinct consecutive frames.

    An odd sampled frame count pads by repeating the last frame, so the final
    temporal patch is full.

    Args:
        frames: The clip's sampled frames, in ascending source order.
        patch_size: Vision patch size.
        merge_size: Spatial merge factor.
        temporal_patch_size: Frames per temporal patch.
        min_pixels: Whole-clip pixel floor.
        max_pixels: Whole-clip pixel ceiling.

    Returns:
        A ``(pixel_values, grid_thw)`` pair, where ``pixel_values`` has shape
        ``[grid_t * grid_h * grid_w, channels * temporal_patch_size *
        patch_size**2]``.

    Raises:
        ValueError: If the clip has no frames, or its frames differ in size.
    """
    if not frames:
        raise ValueError("Video has no frames after decoding.")

    sizes = {frame.size for frame in frames}
    if len(sizes) != 1:
        raise ValueError(
            f"All frames in a clip must share one size, got {sorted(sizes)}."
        )

    width, height = frames[0].size
    (grid_t, grid_h, grid_w), (h_bar, w_bar) = clip_grid(
        len(frames),
        height,
        width,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    resized = [
        frame
        if (h_bar, w_bar) == (height, width)
        else frame.resize((w_bar, h_bar), resample=Image.Resampling.BICUBIC)
        for frame in frames
    ]

    # (T, H, W, C) -> (T, C, H, W), matching the reference's channel-first
    # frames, then the fused rescale-and-normalize the image path uses:
    # (x / 255 - mean) / std with mean = std = 0.5.
    clip = np.stack([np.asarray(frame, dtype=np.float32) for frame in resized])
    clip = clip.transpose(0, 3, 1, 2)
    clip = (clip - (0.5 * 255.0)) / (0.5 * 255.0)

    if pad := -clip.shape[0] % temporal_patch_size:
        clip = np.concatenate([clip, np.repeat(clip[-1:], pad, axis=0)], axis=0)

    channels = clip.shape[1]
    patches = clip.reshape(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # The reference's permute with its batch axis dropped. Identical to the
    # image path's transpose: the row layout is shared, only the temporal
    # content differs.
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

    pixel_values = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    ).astype(np.float32, copy=False)

    return pixel_values, (grid_t, grid_h, grid_w)


def clip_timestamps(
    indices: Sequence[int] | npt.NDArray[np.integer],
    container_fps: float | None,
    *,
    temporal_patch_size: int = 2,
) -> list[float]:
    """Averages each temporal patch's two frame timestamps, in seconds.

    One label per temporal patch, which is what the prompt carries between a
    clip's ``grid_t`` placeholder runs. Qwen wants ``<12.5 seconds>``; gemma4's
    ``MM:SS`` formatting is a different model's format and must not be reused.

    Args:
        indices: The sampled source frame indices.
        container_fps: The container's declared frame rate, or ``None``.
        temporal_patch_size: Frames per temporal patch.

    Returns:
        One timestamp per temporal patch.
    """
    padded = [int(index) for index in indices]
    if remainder := len(padded) % temporal_patch_size:
        padded.extend([padded[-1]] * (temporal_patch_size - remainder))

    effective_fps = (
        container_fps
        if container_fps is not None and container_fps > 0
        else QWEN3VL_DEFAULT_CONTAINER_FPS
    )
    seconds = [index / effective_fps for index in padded]
    return [
        (seconds[i] + seconds[i + temporal_patch_size - 1]) / 2
        for i in range(0, len(seconds), temporal_patch_size)
    ]


def _decode_frames_at(
    video_bytes: bytes, indices: Sequence[int]
) -> tuple[dict[int, Image.Image], int]:
    """Decodes one pass, keeping only the frames at ``indices``.

    Frames outside ``indices`` are dropped as they stream, so peak memory is
    the kept frames plus one in flight rather than the whole clip -- a
    ten-minute 1080p clip is ~100 GB of uncompressed rasters.

    Returns:
        A ``(kept, actual_total)`` pair: the kept frames by index, and how many
        frames the decode actually produced.
    """
    wanted = set(indices)
    kept: dict[int, Image.Image] = {}
    actual = 0
    with open_video_container(io.BytesIO(video_bytes)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in wanted:
                kept[idx] = frame.to_image().convert("RGB")
            actual = idx + 1
    return kept, actual


@traced
def decode_clip(
    video_bytes: bytes,
    *,
    fps: float = QWEN3VL_VIDEO_FPS,
    min_frames: int = QWEN3VL_VIDEO_MIN_FRAMES,
    max_frames: int = QWEN3VL_VIDEO_MAX_FRAMES,
) -> tuple[list[Image.Image], npt.NDArray[np.int64], float | None]:
    """Decodes a clip down to its sampled frames.

    Sampling indices are chosen from the container's declared frame count (or
    a counting decode when the header omits it) so only the sampled frames are
    ever materialized.

    Args:
        video_bytes: The raw encoded clip.
        fps: Frames to sample per second of source video.
        min_frames: Sampling floor, itself capped by the clip's frame count.
        max_frames: Sampling ceiling.

    Returns:
        A ``(frames, indices, container_fps)`` triple. ``container_fps`` is
        ``None`` when the container declares no frame rate; the sampler and the
        timestamps both fall back to the reference's assumed 24.

    Raises:
        ValueError: If the clip holds no decodable frames.
    """
    with open_video_container(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        avg_rate = stream.average_rate
        container_fps = float(avg_rate) if avg_rate else None
        total_frames = stream.frames or 0

    if total_frames <= 0:
        # The header omits a frame count; count with a decode that retains
        # nothing.
        with open_video_container(io.BytesIO(video_bytes)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))

    def sample(total: int) -> npt.NDArray[np.int64]:
        return sample_frame_indices(
            total,
            container_fps,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    indices = sample(total_frames)
    kept, actual_total = _decode_frames_at(video_bytes, indices.tolist())
    if actual_total != total_frames:
        # The container header lied about the frame count; resample from what
        # the decode actually produced.
        indices = sample(actual_total)
        kept, _ = _decode_frames_at(video_bytes, indices.tolist())

    return [kept[i] for i in indices.tolist()], indices, container_fps


@traced
def preprocess_clip(
    video_bytes: bytes,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = QWEN3VL_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3VL_VIDEO_MAX_PIXELS,
    fps: float = QWEN3VL_VIDEO_FPS,
    min_frames: int = QWEN3VL_VIDEO_MIN_FRAMES,
    max_frames: int = QWEN3VL_VIDEO_MAX_FRAMES,
) -> tuple[npt.NDArray[np.float32], tuple[int, int, int], list[float]]:
    """Decodes, samples and patchifies one clip.

    Args:
        video_bytes: The raw encoded clip.
        patch_size: Vision patch size.
        merge_size: Spatial merge factor.
        temporal_patch_size: Frames per temporal patch.
        min_pixels: Whole-clip pixel floor.
        max_pixels: Whole-clip pixel ceiling.
        fps: Frames to sample per second of source video.
        min_frames: Sampling floor, itself capped by the clip's frame count.
        max_frames: Sampling ceiling.

    Returns:
        A ``(pixel_values, grid_thw, timestamps)`` triple, with one timestamp
        per temporal patch -- one per placeholder run the prompt carries.
    """
    frames, indices, container_fps = decode_clip(
        video_bytes, fps=fps, min_frames=min_frames, max_frames=max_frames
    )
    pixel_values, grid_thw = patchify_clip(
        frames,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    timestamps = clip_timestamps(
        indices, container_fps, temporal_patch_size=temporal_patch_size
    )
    return pixel_values, grid_thw, timestamps
