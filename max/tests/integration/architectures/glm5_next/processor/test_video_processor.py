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
"""GLM-5.3-Flash frame sampling, clip decoding and clip patchifying.

The expected index lists were produced by the reference's
``Glm5NextVideoProcessor.sample_frames`` (``modular_glm5_next.py`` at
``f57a815``), one case per branch of its rule.
"""

from __future__ import annotations

import io
import tracemalloc

import av
import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.architectures.glm5_next.video_processor import (
    Glm5NextVideoProcessor,
)

# Verbatim from the checkpoint's processor_config.json, `video_processor` key.
_PROCESSOR_CONFIG = {
    "video_processor": {
        "do_rescale": True,
        "patch_expand_factor": 1,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "patch_size": 14,
        "max_image_tokens": 240000,
        "fps": 2,
    }
}


def _processor() -> Glm5NextVideoProcessor:
    return Glm5NextVideoProcessor.from_processor_config(_PROCESSOR_CONFIG)


def _frame(height: int, width: int, seed: int) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("frames", "native_fps", "duration", "target_fps", "expected"),
    [
        pytest.param(300, 30.0, 10.0, 1.0, list(range(0, 300, 30)), id="walk"),
        # The walk stops short of `extract_t`, so linspace fills the sampled
        # span, then the dedup drops the repeat.
        pytest.param(5, 2.0, 2.5, 2.0, [0, 1, 2, 3], id="walk-too-short"),
        # Fewer frames than `extract_t`: linspace over the clip, dedup, and
        # the last index repeated to an even count.
        pytest.param(3, 1.0, 3.0, 2.0, [0, 1, 2, 2], id="fps-below-target"),
        pytest.param(
            7, 1.0, 7.0, 1.0, [0, 1, 2, 3, 4, 5, 6, 6], id="odd-count"
        ),
        # `extract_t` is 0, so a sub-second clip samples nothing.
        pytest.param(59, 60.0, 0.9, 1.0, [], id="under-one-second"),
    ],
)
def test_sample_frames_matches_reference(
    frames: int,
    native_fps: float,
    duration: float,
    target_fps: float,
    expected: list[int],
) -> None:
    indices = _processor().sample_frames(
        frames, native_fps, duration, target_fps
    )

    assert indices.tolist() == expected


def test_sample_frames_caps_at_max_frames() -> None:
    """The walk overshoots the cap, so linspace spans the whole clip."""
    indices = _processor().sample_frames(240000, 30.0, 8000.0, 1.0)

    assert len(indices) == 2048
    assert indices[:3].tolist() == [0, 117, 234]
    assert indices[-2:].tolist() == [239881, 239999]


def test_sample_frames_cost_follows_the_sample() -> None:
    """A clip's header can declare any frame count; sampling must not walk it.

    The reference builds a timestamp per declared frame, about 35 MB for a
    million; the 2048 frames kept here take a small fraction of that.
    """
    tracemalloc.start()
    try:
        indices = _processor().sample_frames(10**6, 30.0, 10**6 / 30.0, 2.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(indices) == 2048
    assert indices[-1] == 10**6 - 1
    assert peak < 1 << 20


def test_missing_fps_is_refused() -> None:
    with pytest.raises(ValueError, match="native fps"):
        _processor().sample_frames(100, 0.0, 10.0, 2.0)


def test_clip_grid_matches_patchified_output() -> None:
    """``grid_thw`` must predict what ``preprocess`` produces."""
    processor = _processor()
    clip = [_frame(360, 640, seed) for seed in range(4)]

    patches, grid = processor.preprocess(clip)

    assert processor.grid_thw(clip) == grid
    assert grid[0] == 2
    assert patches.shape[0] == grid[0] * grid[1] * grid[2]
    assert patches.shape[1] == 3 * 2 * 14 * 14


def test_temporal_patch_fits_one_encode_pass() -> None:
    """A 4K frame pair is shrunk to what one encode pass holds.

    Only the clip-wide budget bounds a short clip's canvas, and four 4K frames
    fit it at 10,764 tokens a pair. A clip already within one pass keeps the
    reference's canvas.
    """
    uncapped = _processor()
    capped = Glm5NextVideoProcessor.from_processor_config(
        _PROCESSOR_CONFIG, max_entry_tokens=8192
    )

    def tokens(canvas: tuple[int, int]) -> int:
        return canvas[0] * canvas[1] // 28**2

    assert tokens(uncapped.canvas_size(2160, 3840, 4)) == 10764
    assert tokens(capped.canvas_size(2160, 3840, 4)) <= 8192
    assert capped.canvas_size(720, 1280, 4) == uncapped.canvas_size(
        720, 1280, 4
    )


def test_video_budget_is_the_video_one() -> None:
    """Read from the `video_processor` section, not the image one."""
    processor = _processor()

    assert processor.max_image_tokens == 240000
    assert processor.fps == 2
    assert processor.max_frames == 2048


def _mp4(num_frames: int, fps: int, width: int = 32, height: int = 32) -> bytes:
    """An H.264 clip whose frame ``i`` is a flat ``8 * i`` grey."""
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    assert isinstance(stream, av.VideoStream)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for index in range(num_frames):
        pixels = np.full((height, width, 3), 8 * index, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def _grey_levels(frames: list[npt.NDArray[np.uint8]]) -> list[int]:
    return [round(float(frame.mean()) / 8) for frame in frames]


def test_decode_fetches_the_frames_sample_frames_picks() -> None:
    """Seven frames at 1 fps: an odd sample, so the last frame is repeated."""
    frames, indices, fps = _processor().decode(_mp4(7, 1))

    assert indices == [0, 1, 2, 3, 4, 5, 6, 6]
    assert fps == 1.0
    assert _grey_levels(frames) == indices


def test_decode_keeps_the_first_frame_of_a_sub_second_clip() -> None:
    """0.4 s at 2 fps: the reference samples nothing, and MAX one pair.

    The reference's processor then divides by the empty grid, so there is no
    behaviour of its to match; a clip this short is one temporal patch of its
    first frame rather than a failed request.
    """
    frames, indices, _ = _processor().decode(_mp4(4, 10))

    assert indices == [0, 0]
    assert _grey_levels(frames) == [0, 0]
