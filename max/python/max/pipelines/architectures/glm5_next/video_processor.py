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
"""GLM-5.3-Flash video frame sampling and preprocessing.

The pixel pipeline is :class:`~.image_processor.Glm5NextPatchifier`'s -- same
``smart_resize``, same resize-then-pad canvas, same patchify -- with a
240,000-token budget instead of 8,000 and a real temporal extent instead of a
still image duplicated into the temporal patch.

:meth:`Glm5NextVideoProcessor.sample_frames` is the new work, and it is not
GLM-4.6V's rule. Four things about it are easy to get subtly wrong, and each
produces a clip that encodes and answers slightly badly rather than failing:

* the second-boundary walk advances by ``1 / target_fps`` but breaks against
  ``int(duration)``, so it yields about ``duration * target_fps`` frames and the
  cap is on the *original* duration, not a clamped one;
* ``np.linspace`` is the fallback in both directions -- too few frames and too
  many -- but over different ranges: the sampled span in the first case and the
  whole clip in the second;
* duplicates are removed *after* the fallback, so the count can drop below
  ``extract_t``;
* the even-count forcing runs *after* the dedup and duplicates the last index,
  so the returned list can legitimately repeat. The patch embedding consumes
  frames in temporal pairs, so an odd count would leave a half-filled temporal
  patch.

Budgets are the shipped ones: ``max_image_tokens`` 240,000 decoder tokens,
``fps`` 2, ``max_frames`` 2048. ``max_image_size`` (``28 * 28 * 2 * 30000``)
exists in the reference's config but its resize path does not read it, so it is
not modelled here.
"""

from __future__ import annotations

import logging
import math
import mmap
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import VideoSource, decode_video_frames
from PIL import Image
from typing_extensions import Self

from .image_processor import Glm5NextPatchifier, smart_resize

__all__ = ["Glm5NextVideoProcessor"]

logger = logging.getLogger("max.pipelines")

# The reference processor's frame rate for a clip that declares none.
_DEFAULT_FPS = 24.0

# Workers that resize and patchify one clip's temporal patches.
_PREPROCESS_THREADS = 16


def _second_boundary_walk(
    total_num_frames: int,
    native_fps: float,
    target_fps: float,
    max_seconds: int,
    limit: int,
) -> list[int]:
    """The reference's second-boundary walk, at most ``limit`` steps long.

    The reference scans every frame's timestamp, ``i * (1 / native_fps)``, for
    the first at or past a running sum of ``1 / target_fps``. This jumps to
    that frame instead, so the cost follows the frames kept rather than the
    clip's declared frame count, with the reference's float arithmetic.
    """
    duration_per_frame = 1 / native_fps
    inv_fps = 1 / target_fps
    indices: list[int] = []
    current_second = 0.0
    lowest = 0
    while lowest < total_num_frames and len(indices) < limit:
        index = max(lowest, math.ceil(current_second / duration_per_frame))
        # The division can round across a frame boundary either way.
        while (
            index > lowest
            and (index - 1) * duration_per_frame >= current_second
        ):
            index -= 1
        while (
            index < total_num_frames
            and index * duration_per_frame < current_second
        ):
            index += 1
        if index >= total_num_frames:
            break
        current_second += inv_fps
        indices.append(index)
        if current_second >= max_seconds:
            break
        lowest = index + 1
    return indices


class Glm5NextVideoProcessor(Glm5NextPatchifier):
    """Samples frames from a clip and patchifies them for the GLM-5.3 tower."""

    CONFIG_SECTION = "video_processor"

    def __init__(
        self,
        *,
        max_image_tokens: int = 240000,
        fps: float = 2.0,
        max_frames: int = 2048,
        max_entry_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_image_tokens=max_image_tokens, **kwargs)
        self.fps = fps
        self.max_frames = max_frames
        self.max_entry_tokens = max_entry_tokens

    @classmethod
    def from_processor_config(
        cls, config: Mapping[str, Any], **overrides: Any
    ) -> Self:
        section = config.get(cls.CONFIG_SECTION, config)
        defaults = cls()
        overrides.setdefault("fps", section.get("fps", defaults.fps))
        overrides.setdefault(
            "max_frames", section.get("max_frames", defaults.max_frames)
        )
        return super().from_processor_config(config, **overrides)

    def sample_frames(
        self,
        total_num_frames: int,
        native_fps: float,
        duration: float | None = None,
        target_fps: float | None = None,
        max_frames: int | None = None,
    ) -> npt.NDArray[np.int64]:
        """Picks which frame indices of a clip the tower sees.

        Args:
            total_num_frames: Frames the decoded clip holds.
            native_fps: The clip's own frame rate, from its metadata.
            duration: The clip's duration in seconds, if the container reports
                one. Derived from the frame count and rate when it does not.
            target_fps: Sampling rate. Defaults to the configured ``fps``.
            max_frames: Frame cap. Defaults to the configured ``max_frames``;
                a smaller one, such as a decoded-memory budget's, samples as
                a checkpoint configured with it would.

        Returns:
            An even-length, ascending array of frame indices, possibly with the
            last index repeated once.

        Raises:
            ValueError: If ``native_fps`` is missing, which the reference also
                refuses to guess at.
        """
        if not native_fps > 0:
            raise ValueError(
                "Frame sampling needs the clip's native fps; GLM-5.3-Flash "
                "budgets frames in seconds, not in frame counts."
            )
        max_frame_idx = total_num_frames - 1
        duration = duration or round(max_frame_idx / native_fps) + 1
        # The cap is on the original duration. The reference computes it before
        # any `max_duration` clamp, which is disabled in the shipped config.
        max_seconds = int(duration)
        if target_fps is None:
            target_fps = self.fps
        if max_frames is None:
            max_frames = self.max_frames

        extract_t = min(int(duration * target_fps), max_frames)

        frame_indices: list[int]
        if total_num_frames < extract_t:
            frame_indices = np.linspace(
                0, total_num_frames - 1, extract_t, dtype=int
            ).tolist()
        else:
            # A walk longer than `extract_t` is replaced below whatever it
            # holds, so one past it is all the walk needs to find.
            frame_indices = _second_boundary_walk(
                total_num_frames,
                native_fps,
                target_fps,
                max_seconds,
                limit=extract_t + 1,
            )

        if len(frame_indices) < extract_t:
            if not frame_indices:
                start, end = 0, max(total_num_frames - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(
                start, end, extract_t, dtype=int
            ).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(
                0, total_num_frames - 1, extract_t, dtype=int
            ).tolist()

        seen: set[int] = set()
        unique: list[int] = []
        for index in frame_indices:
            if index not in seen:
                seen.add(index)
                unique.append(index)

        # After the dedup, and by duplicating the last index -- so the result
        # can repeat a frame. The patch embedding pairs frames temporally.
        if len(unique) & 1:
            unique.append(unique[-1])

        return np.array(unique, dtype=np.int64)

    def decode(
        self, source: VideoSource, max_decoded_bytes: int | None = None
    ) -> tuple[list[npt.NDArray[np.uint8]], list[int], float]:
        """Decodes the frames :meth:`sample_frames` picks from a clip.

        The duration is the frame count over the frame rate, as transformers'
        PyAV loader computes it. A decoded-memory budget lowers the frame cap
        rather than rejecting the clip.

        A clip too short to yield a frame at the target rate -- under half a
        second at 2 fps -- gets its first frame as one temporal pair. There is
        no reference behaviour to match: the reference samples no frames and
        its processor then divides by the empty grid.

        Args:
            source: The encoded clip.
            max_decoded_bytes: Ceiling on the decoded frames' footprint.

        Returns:
            The sampled frames as ``(height, width, 3)`` uint8 RGB arrays,
            their source indices, and the frame rate the indices count in,
            which the timestamps in the prompt are built from. The frames
            stay the arrays the budget sized: converting the clip to Pillow
            images up front, at four bytes an RGB pixel, would briefly hold
            7/3 of it.
        """

        def sample(total: int, fps: float, max_frames: int | None) -> list[int]:
            wanted = min(int(total / fps * self.fps), self.max_frames)
            if max_frames is not None and max_frames < wanted:
                logger.warning(
                    "Sampling %d of the %d frames GLM-5.3-Flash takes from a "
                    "video; the rest do not fit the request's media budget "
                    "(MAX_SERVE_MAX_MEDIA_BYTES).",
                    max_frames,
                    wanted,
                )
            # The sampler pads an odd count with a repeated frame, so an odd
            # cap would keep one frame more than the budget allows.
            if max_frames is not None and max_frames > 1:
                max_frames -= max_frames % 2
            indices = self.sample_frames(
                total, fps, duration=total / fps, max_frames=max_frames
            )
            return indices.tolist() or [0, 0]

        sampled = decode_video_frames(
            source,
            sample_indices=sample,
            max_frames=self.max_frames,
            max_decoded_bytes=max_decoded_bytes,
            default_fps=_DEFAULT_FPS,
        )
        return sampled.frames, sampled.indices, sampled.info.fps or _DEFAULT_FPS

    def canvas_size(
        self, height: int, width: int, num_frames: int | None = None
    ) -> tuple[int, int]:
        """Returns the clip's canvas, shrunk until one temporal patch fits.

        The model encodes each temporal patch in one pass and never splits
        one, so a patch over ``max_entry_tokens`` gets the largest canvas that
        holds that many tokens. Any other clip keeps the reference's canvas.
        """
        canvas_height, canvas_width = super().canvas_size(
            height, width, num_frames
        )
        if (
            self.max_entry_tokens is None
            or canvas_height * canvas_width // self.spatial_factor**2
            <= self.max_entry_tokens
        ):
            return canvas_height, canvas_width
        return smart_resize(
            self.temporal_patch_size,
            height,
            width,
            temporal_factor=self.temporal_patch_size,
            factor=self.spatial_factor,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_entry_tokens,
        )

    def grid_thw(
        self, frames: Sequence[npt.NDArray[np.uint8]]
    ) -> tuple[int, int, int]:
        """Returns a clip's ``(t, h, w)`` grid without touching the pixels."""
        height, width = frames[0].shape[:2]
        num_frames = len(frames)
        canvas_height, canvas_width = self.canvas_size(
            height, width, num_frames
        )
        remainder = num_frames % self.temporal_patch_size
        aligned_frames = num_frames + (
            self.temporal_patch_size - remainder if remainder else 0
        )
        return (
            aligned_frames // self.temporal_patch_size,
            canvas_height // self.patch_size,
            canvas_width // self.patch_size,
        )

    def preprocess(
        self, frames: Sequence[npt.NDArray[np.uint8]]
    ) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
        """Patchifies an already-sampled clip.

        Every frame shares one canvas, sized from the frames' common size and
        the clip's frame count, because the tower consumes a single
        ``(t, h, w)`` grid per clip.

        Args:
            frames: The sampled ``(height, width, 3)`` uint8 RGB frames, in
                order. Each is a Pillow image only while it is resized.

        Returns:
            The ``(num_patches, 3 * temporal_patch_size * patch_size**2)``
            float32 patch tensor and its ``(t, h, w)`` grid.

        Raises:
            ValueError: If ``frames`` is empty or the frames disagree on size.
        """
        if not frames:
            raise ValueError("A clip needs at least one frame.")
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(
                f"All frames of a clip must share one size, got {shapes}."
            )

        height, width = frames[0].shape[:2]
        num_frames = len(frames)
        canvas_height, canvas_width = self.canvas_size(
            height, width, num_frames
        )
        content_height, content_width = self.content_size(
            height, width, canvas_height, canvas_width, num_frames
        )
        step = self.temporal_patch_size
        grid_t = -(-num_frames // step)
        grid_h = canvas_height // self.patch_size
        grid_w = canvas_width // self.patch_size
        rows = grid_h * grid_w
        # Anonymous pages rather than NumPy's: NumPy advises huge pages for a
        # large array, and where a big page cache has fragmented memory each
        # huge-page fault compacts synchronously, which costs minutes on a
        # clip's 4.5 GB against about a second of small-page faults.
        row_size = 3 * step * self.patch_size**2
        patches = np.frombuffer(
            mmap.mmap(-1, grid_t * rows * row_size * np.float32().itemsize),
            dtype=np.float32,
        ).reshape(grid_t * rows, row_size)

        canvases = threading.local()

        def patchify_temporal_patch(t: int) -> None:
            if not hasattr(canvases, "padded"):
                canvases.padded = np.zeros(
                    (step, canvas_height, canvas_width, 3), dtype=np.uint8
                )
            pair = frames[t * step : (t + 1) * step]
            for frame, canvas in zip(pair, canvases.padded, strict=False):
                self.resize_and_pad(
                    Image.fromarray(frame),
                    canvas_height,
                    canvas_width,
                    content_height,
                    content_width,
                    out=canvas,
                )
            self.patchify(
                canvases.padded[: len(pair)],
                out=patches[t * rows : (t + 1) * rows],
            )

        # A long clip is hundreds of frames, and both the resize and the
        # patchify release the GIL. Each worker reuses one canvas: under
        # jemalloc a fresh zeroed one per frame is a page fault per 4 KiB, and
        # the faults of many threads contend.
        with ThreadPoolExecutor(
            max_workers=min(_PREPROCESS_THREADS, grid_t)
        ) as pool:
            list(pool.map(patchify_temporal_patch, range(grid_t)))
        return patches, (grid_t, grid_h, grid_w)

    def __call__(
        self, clips: Sequence[Sequence[npt.NDArray[np.uint8]]]
    ) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.int32]]:
        """Patchifies each clip, returning per-clip patches and the grids."""
        processed = [self.preprocess(clip) for clip in clips]
        return (
            [patches for patches, _ in processed],
            np.array([grid for _, grid in processed], dtype=np.int32).reshape(
                len(processed), 3
            ),
        )
