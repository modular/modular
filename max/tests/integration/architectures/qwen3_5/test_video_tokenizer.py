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

"""Pins how a clip becomes a prompt and a set of vision encoder inputs.

Three contracts meet in ``Qwen3VLTokenizer.new_context`` and none of them is
visible from the image path:

- A clip is ONE encoder grid item with ``t > 1`` but ``grid_t`` SEPARATE
  placeholder runs in the prompt, each preceded by a timestamp label. So its
  ``ImageMetadata`` span is wider than the rows the encoder emits for it, and
  ``num_embedding_rows`` is what reconciles the two.
- The bilinear position arrays are ``sum(h * w)`` while the pixels are
  ``sum(t * h * w)``. Asserting that asymmetry here is what keeps the vision
  graph's two symbolic dimensions honest.
- ``ctx.images`` and the encoder's grid rows must stay 1:1 in PROMPT order,
  because ``vision_packing._split_vision_data`` zips them. Images and clips
  interleave freely, so this is not the same as images-then-clips.

The still-image assertions are the regression half: they recompute what the
image path produced before video existed and require it byte for byte.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from unittest.mock import MagicMock, NonCallableMock

import av
import numpy as np
import pytest
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.architectures.qwen3_5.vision_packing import (
    _split_vision_data,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.architectures.qwen3vl_moe.nn.video_processing import (
    clip_grid,
)
from max.pipelines.architectures.qwen3vl_moe.tokenizer import (
    VIDEO_PAD_TOKEN,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.modeling.types import (
    ImageContentPart,
    MessageContent,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VideoContentPart,
)
from max.support.image import find_contiguous_ranges
from PIL import Image
from transformers import AutoConfig

# Config + tokenizer files only; the weights are never loaded.
MODEL_PATH = "Qwen/Qwen3.5-9B"

MERGE_SIZE = 2
PATCH_ROW_WIDTH = 3 * 2 * 16 * 16


def _mock_pipeline_config(model_path: str) -> MagicMock:
    """A PipelineConfig stand-in wrapping the real HF config.

    ``Qwen3VLTokenizer`` reads the vision config and the vision token IDs
    straight off it, so the config has to be real; everything else is mocked.
    """
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.enable_prefix_caching = False

    mock_model_config = MagicMock()
    mock_model_config.huggingface_config = hf_config
    mock_model_config.kv_cache = mock_kv_cache_config

    pipeline_config = MagicMock()
    pipeline_config.model = mock_model_config
    pipeline_config.runtime.vision_cache_utilization = 0
    pipeline_config.runtime.max_vision_preprocess_cache_bytes = 0
    pipeline_config.runtime.max_video_preprocess_cache_bytes = 0
    pipeline_config.runtime.max_media_preprocess_cache_idle_seconds = 0.0
    return pipeline_config


@pytest.fixture(scope="module")
def tokenizer() -> Qwen3_5Tokenizer:
    return Qwen3_5Tokenizer(
        model_path=MODEL_PATH,
        pipeline_config=_mock_pipeline_config(MODEL_PATH),
        trust_remote_code=True,
    )


def _png(size: tuple[int, int], value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def _mp4(size: tuple[int, int], num_frames: int, fps: int) -> bytes:
    """An H.264 clip of ``num_frames`` distinguishable frames."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        assert isinstance(stream, av.video.stream.VideoStream)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(num_frames):
            frame = Image.new("RGB", size, (4 * i, 255 - 4 * i, 128))
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _context(
    tokenizer: Qwen3_5Tokenizer,
    content: Sequence[MessageContent],
    *,
    images: Sequence[bytes] = (),
    videos: Sequence[bytes] = (),
) -> Qwen3VLTextAndVisionContext:
    request = TextGenerationRequest(
        request_id=RequestID("test-video-tokenizer"),
        model_name=MODEL_PATH,
        messages=[
            TextGenerationRequestMessage(role="user", content=list(content))
        ],
        images=list(images),
        videos=list(videos),
    )
    return asyncio.run(tokenizer.new_context(request))


def _runs(
    ctx: Qwen3VLTextAndVisionContext, token_id: int
) -> list[tuple[int, int]]:
    return find_contiguous_ranges(np.asarray(ctx.tokens.all), [token_id])


# --------------------------------------------------------------------------
# Still images: the regression half.
# --------------------------------------------------------------------------


def test_image_only_context_is_unchanged(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """An image-only request produces exactly what it did before video.

    Every quantity below is recomputed from the image path's own rules rather
    than compared against a recorded blob, so a drift in either the grid or the
    ordering fails here.
    """
    images = [_png((256, 384), 30), _png((384, 256), 200)]
    ctx = _context(
        tokenizer,
        [
            TextContentPart(text="Compare these."),
            ImageContentPart(),
            ImageContentPart(),
        ],
        images=images,
    )

    assert ctx.vision_data is not None
    grids = np.asarray(ctx.vision_data.image_grid_thw)
    # No clip, so no temporal extent and no video grid at all.
    np.testing.assert_array_equal(grids[:, 0], [1, 1])
    assert ctx.vision_data.video_grid_thw is None

    tokens = np.asarray(ctx.tokens.all)
    # The prompt carries no video placeholder, so widening
    # `image_token_indices` to both modalities cannot have changed it.
    assert not (tokens == tokenizer.video_token_id).any()
    np.testing.assert_array_equal(
        ctx.image_token_indices,
        (tokens == tokenizer.image_token_id).nonzero()[0],
    )

    spans = _runs(ctx, tokenizer.image_token_id)
    assert len(ctx.images) == len(spans) == 2
    for img, (start, end), grid in zip(ctx.images, spans, grids, strict=True):
        assert (img.start_idx, img.end_idx) == (start, end)
        # An image's span is all placeholders, so no override is needed and
        # the derived row count still equals the span width.
        assert img.num_embedding_rows is None
        assert img.embedding_rows == end - start
        assert end - start == int(np.prod(grid)) // MERGE_SIZE**2

    # Position arrays and pixels have the SAME length for images -- which is
    # exactly why nothing here could ever have caught the clip asymmetry.
    patch_rows = int(np.prod(grids, axis=1).sum())
    plane_rows = int((grids[:, 1] * grids[:, 2]).sum())
    assert patch_rows == plane_rows
    assert ctx.vision_data.concatenated_pixel_values.shape == (
        patch_rows,
        PATCH_ROW_WIDTH,
    )
    assert ctx.vision_data.vision_position_ids.shape == (patch_rows, 2)
    assert ctx.vision_data.indices.shape == (4, plane_rows)
    assert ctx.vision_data.weights.shape == (4, plane_rows, 1)


def test_image_pixels_are_concatenated_in_prompt_order(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """Two differently-sized images keep prompt order through the split.

    The order is load-bearing rather than cosmetic: ``pack_uncached_images``
    zips ``ctx.images`` against this concatenation's cumulative-sum walk.
    """
    ctx = _context(
        tokenizer,
        [ImageContentPart(), TextContentPart(text="and"), ImageContentPart()],
        images=[_png((256, 384), 30), _png((384, 256), 200)],
    )
    assert ctx.vision_data is not None
    pieces = _split_vision_data(ctx)
    assert len(pieces) == 2
    # (256, 384) is w x h, so the first grid is taller than wide and the
    # second is the reverse. Swapped order would swap these. Both are above
    # the image path's 65536-pixel floor, so neither is rescaled.
    assert tuple(pieces[0].grid[1:]) == (24, 16)
    assert tuple(pieces[1].grid[1:]) == (16, 24)


# --------------------------------------------------------------------------
# Clips.
# --------------------------------------------------------------------------


def _expected_clip_grid(
    size: tuple[int, int], num_frames: int, fps: int
) -> tuple[int, int, int]:
    """The grid the sampler + resize should produce for ``_mp4``'s clip."""
    width, height = size
    sampled = int(num_frames / fps * 2)
    grid, _ = clip_grid(sampled, height, width)
    return grid


def test_clip_becomes_grid_t_placeholder_runs(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """One clip, ``grid_t`` runs, one timestamp label before each."""
    size, num_frames, fps = (96, 64), 48, 12
    video = _mp4(size, num_frames, fps)
    grid_t, grid_h, grid_w = _expected_clip_grid(size, num_frames, fps)
    assert grid_t > 1, "the fixture must exercise more than one temporal patch"

    ctx = _context(
        tokenizer,
        [TextContentPart(text="Describe this."), VideoContentPart()],
        videos=[video],
    )

    runs = _runs(ctx, tokenizer.video_token_id)
    assert len(runs) == grid_t
    tokens_per_run = grid_h * grid_w // MERGE_SIZE**2
    for start, end in runs:
        assert end - start == tokens_per_run

    prompt = tokenizer.delegate.decode(
        ctx.tokens.all.tolist(), skip_special_tokens=False
    )
    # One label per run, each the mean of its temporal patch's two frame
    # timestamps at one decimal. 48 frames at 12 fps samples 8 of them --
    # linspace(0, 47, 8).round() -- so the labels are these and not the frame
    # times, and not `MM:SS` as gemma4 would format them.
    assert prompt.count(" seconds>") == grid_t
    for label in (
        "<0.3 seconds>",
        "<1.4 seconds>",
        "<2.5 seconds>",
        "<3.6 seconds>",
    ):
        assert label in prompt, prompt
    # Each run gets its own vision-start/end pair, nested inside the single
    # pair the chat template emitted around the whole clip.
    assert prompt.count("<|vision_start|>") == grid_t + 1
    assert prompt.count("<|vision_end|>") == grid_t + 1


def test_clip_span_covers_its_runs_and_declares_fewer_rows(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """The clip's span is wider than the embeddings it consumes.

    A span made of pure placeholders (an image) needs no override. A clip's
    span swallows the timestamp text between its runs, so
    ``num_embedding_rows`` is what stops the driver deriving a row count from
    the span width -- which would over-count by the label tokens.
    """
    size, num_frames, fps = (96, 64), 48, 12
    grid_t, grid_h, grid_w = _expected_clip_grid(size, num_frames, fps)

    ctx = _context(
        tokenizer,
        [VideoContentPart()],
        videos=[_mp4(size, num_frames, fps)],
    )

    assert len(ctx.images) == 1
    clip = ctx.images[0]
    runs = _runs(ctx, tokenizer.video_token_id)
    assert (clip.start_idx, clip.end_idx) == (runs[0][0], runs[-1][1])

    rows = grid_t * grid_h * grid_w // MERGE_SIZE**2
    assert clip.num_embedding_rows == rows
    assert clip.embedding_rows == rows
    assert clip.end_idx - clip.start_idx > rows, (
        "the span must include the timestamp text, or this test proves nothing"
    )

    # The scatter walks `image_token_indices` from the span's start for
    # exactly `embedding_rows` positions, so those must be the clip's runs.
    positions = np.asarray(ctx.image_token_indices)
    assert len(positions) == rows
    np.testing.assert_array_equal(
        positions,
        np.concatenate([np.arange(lo, hi) for lo, hi in runs]),
    )


def test_clip_position_arrays_are_shorter_than_its_pixels(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """The asymmetry the vision graph's two symbolic dimensions exist for."""
    size, num_frames, fps = (96, 64), 48, 12
    grid_t, grid_h, grid_w = _expected_clip_grid(size, num_frames, fps)

    ctx = _context(
        tokenizer,
        [VideoContentPart()],
        videos=[_mp4(size, num_frames, fps)],
    )

    assert ctx.vision_data is not None
    data = ctx.vision_data
    np.testing.assert_array_equal(
        data.image_grid_thw, [[grid_t, grid_h, grid_w]]
    )
    np.testing.assert_array_equal(
        data.video_grid_thw, [[grid_t, grid_h, grid_w]]
    )

    patch_rows = grid_t * grid_h * grid_w
    plane_rows = grid_h * grid_w
    assert patch_rows == grid_t * plane_rows
    assert data.concatenated_pixel_values.shape == (
        patch_rows,
        PATCH_ROW_WIDTH,
    )
    # Tiled by t, so a patch-row array.
    assert data.vision_position_ids.shape == (patch_rows, 2)
    # Interpolated once per frame, so a plane array -- shorter.
    assert data.indices.shape == (4, plane_rows)
    assert data.weights.shape == (4, plane_rows, 1)

    # Attention runs within each temporal patch, not across them.
    np.testing.assert_array_equal(
        data.cu_seqlens, np.arange(grid_t + 1) * plane_rows
    )
    assert int(data.max_seqlen) == plane_rows


def test_clip_and_images_interleave_in_prompt_order(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """A clip between two images lands in the middle of the encoder's grid.

    Concatenating images-then-clips would put the clip's grid row last while
    its span sat in the middle, and every array the packer splits would be
    misaligned from that row onward.
    """
    size, num_frames, fps = (96, 64), 48, 12
    grid_t, grid_h, grid_w = _expected_clip_grid(size, num_frames, fps)

    ctx = _context(
        tokenizer,
        [
            ImageContentPart(),
            TextContentPart(text="then"),
            VideoContentPart(),
            TextContentPart(text="then"),
            ImageContentPart(),
        ],
        images=[_png((256, 384), 30), _png((384, 256), 200)],
        videos=[_mp4(size, num_frames, fps)],
    )

    assert ctx.vision_data is not None
    grids = np.asarray(ctx.vision_data.image_grid_thw)
    assert len(ctx.images) == 3
    # The clip is the middle grid row, and it is the only one with t > 1.
    np.testing.assert_array_equal(grids[:, 0], [1, grid_t, 1])
    np.testing.assert_array_equal(grids[1], [grid_t, grid_h, grid_w])
    assert [img.start_idx for img in ctx.images] == sorted(
        img.start_idx for img in ctx.images
    )

    # The packer's split must hand each item its own rows.
    pieces = _split_vision_data(ctx)
    assert len(pieces) == 3
    for img, piece, grid in zip(ctx.images, pieces, grids, strict=True):
        t, h, w = (int(v) for v in grid)
        assert piece.pixel_values.shape == (t * h * w, PATCH_ROW_WIDTH)
        assert piece.vision_position_ids.shape == (t * h * w, 2)
        assert piece.weights.shape == (4, h * w, 1)
        assert piece.indices.shape == (4, h * w)
        assert img.embedding_rows == t * h * w // MERGE_SIZE**2

    # The clip's pixels in the concatenation are its own, byte for byte.
    np.testing.assert_array_equal(
        pieces[1].pixel_values,
        ctx.vision_data.concatenated_pixel_values[
            pieces[0].pixel_values.shape[0] : pieces[0].pixel_values.shape[0]
            + grid_t * grid_h * grid_w
        ],
    )


def test_clip_temporal_slots_survive_the_tokenizer(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """The two temporal slots of a row still hold two distinct frames.

    The same property :mod:`test_video_processing` pins on the preprocessor,
    re-asserted after the tokenizer's concatenation and dtype handling, so a
    reshape here cannot silently collapse it.
    """
    ctx = _context(
        tokenizer,
        [VideoContentPart()],
        videos=[_mp4((96, 64), 48, 12)],
    )
    assert ctx.vision_data is not None
    row = ctx.vision_data.concatenated_pixel_values[0].reshape(3, 2, 16, 16)
    assert not np.array_equal(row[:, 0], row[:, 1])


def test_clip_rope_positions_follow_timestamp_semantics(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """Each run is positioned as a still image of ``(h / 2, w / 2)``.

    ``get_rope_index`` expands the clip's ``(grid_t, h, w)`` into ``grid_t``
    rows of ``(1, h, w)``, so the temporal axis carries each run's flat start
    offset instead of a frame index -- the temporal progression lives in the
    timestamp text between the runs.
    """
    size, num_frames, fps = (96, 64), 48, 12
    grid_t, grid_h, grid_w = _expected_clip_grid(size, num_frames, fps)

    ctx = _context(
        tokenizer,
        [VideoContentPart()],
        videos=[_mp4(size, num_frames, fps)],
    )
    positions = np.asarray(ctx.decoder_position_ids)
    assert positions.shape == (3, len(ctx.tokens.all))

    runs = _runs(ctx, tokenizer.video_token_id)
    llm_h, llm_w = grid_h // MERGE_SIZE, grid_w // MERGE_SIZE
    for start, end in runs:
        temporal, height, width = positions[:, start:end]
        # Positions are absolute, so each axis is the run's flat start offset
        # plus its own index. The temporal axis gets no index at all -- that is
        # what `llm_grid_t = 1` means.
        base = int(temporal[0])
        np.testing.assert_array_equal(temporal, np.full(end - start, base))
        np.testing.assert_array_equal(
            height, base + np.repeat(np.arange(llm_h), llm_w)
        )
        np.testing.assert_array_equal(
            width, base + np.tile(np.arange(llm_w), llm_h)
        )
    # Runs advance by their own width, so a clip costs grid_t * llm_h * llm_w
    # position steps rather than one.
    starts = [int(positions[0, start]) for start, _ in runs]
    assert starts == sorted(starts)
    assert len(set(starts)) == grid_t


# --------------------------------------------------------------------------
# Rejections.
# --------------------------------------------------------------------------


def test_clip_below_one_factor_is_rejected(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """A clip whose frames are under 32 px raises rather than being clamped up.

    The image ``smart_resize`` clamps a small side up to one factor; the video
    one raises. Reachable from a real request, unlike a clip without messages
    (which ``TextGenerationRequest`` already rejects).
    """
    with pytest.raises(ValueError, match="must be larger than"):
        _context(
            tokenizer,
            [VideoContentPart()],
            videos=[_mp4((32, 16), 8, 12)],
        )


def test_more_video_placeholders_than_videos_is_rejected(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """A user-injected clip placeholder cannot be silently absorbed."""
    request = TextGenerationRequest(
        request_id=RequestID("test-video-extra-placeholder"),
        model_name=MODEL_PATH,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    VideoContentPart(),
                    TextContentPart(text="<|video_pad|>"),
                ],
            )
        ],
        videos=[_mp4((96, 64), 8, 12)],
    )
    with pytest.raises(ValueError, match="More <\\|video_pad\\|> tokens"):
        asyncio.run(tokenizer.new_context(request))


def test_video_placeholder_without_a_video_is_rejected(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """A literal video placeholder with no video attached is a client error.

    ``image_token_indices`` indexes BOTH modalities, so such a token claims a
    scatter position with no embedding row behind it. The run check has always
    been able to catch this -- it expects zero runs when the request carries no
    clips -- but it only runs if it is called on the empty-clip path too.
    """
    with pytest.raises(ValueError, match="Clip placeholder mismatch"):
        _context(
            tokenizer,
            [TextContentPart(text=f"look: {VIDEO_PAD_TOKEN}")],
        )


def test_video_placeholder_beside_an_image_is_rejected(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """The same, with an image present, which is the damaging case.

    With an entry in the request the indices are built rather than left empty,
    so the stray placeholder would reach the scatter as a real position.
    """
    with pytest.raises(ValueError, match="Clip placeholder mismatch"):
        _context(
            tokenizer,
            [
                TextContentPart(text=f"stray {VIDEO_PAD_TOKEN}"),
                ImageContentPart(),
            ],
            images=[_png((256, 256), 30)],
        )
