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
"""Tests for Gemma4 batched-vision buffer merging."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    build_image_inputs,
    build_video_inputs,
    create_empty_embeddings,
    merge_per_device_buffers,
)
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.context import ImageMetadata
from max.pipelines.context.context import TokenBuffer
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.pipelines.request import RequestID

_HIDDEN = 4


def _buf(rows: int) -> Buffer:
    """A rank-2 [rows, _HIDDEN] CPU buffer with distinct, recoverable values."""
    data = np.arange(rows * _HIDDEN, dtype=np.float32).reshape(rows, _HIDDEN)
    return Buffer.from_numpy(data).to(CPU())


def test_merge_concatenates_rank2_buffers() -> None:
    # Regression test: the on-device concat indexes ``combined`` with a leading
    # slice (``combined[:a_rows, :]``). Indexing a rank-2 buffer with a single
    # index raised "the provided number of indices (1) is not equal to the
    # tensor rank (2)" and crashed the model worker on any batch with 2+ images.
    a, b = _buf(2), _buf(3)
    a_np, b_np = a.to_numpy().copy(), b.to_numpy().copy()

    [merged] = merge_per_device_buffers([a], [b])

    out = merged.to_numpy()
    assert out.shape == (5, _HIDDEN)
    np.testing.assert_array_equal(out[:2], a_np)
    np.testing.assert_array_equal(out[2:], b_np)


def _buf1d(rows: int) -> Buffer:
    """A rank-1 [rows] CPU buffer (the scatter-index shape)."""
    data = np.arange(rows, dtype=np.int32)
    return Buffer.from_numpy(data).to(CPU())


def test_merge_concatenates_rank1_buffers() -> None:
    # Scatter indices are rank-1; regression for the rank-2-only slice that
    # crashed this caller with "indices (2) != tensor rank (1)".
    a, b = _buf1d(2), _buf1d(3)
    a_np, b_np = a.to_numpy().copy(), b.to_numpy().copy()

    [merged] = merge_per_device_buffers([a], [b])

    out = merged.to_numpy()
    assert out.shape == (5,)
    np.testing.assert_array_equal(out[:2], a_np)
    np.testing.assert_array_equal(out[2:], b_np)


def test_merge_is_elementwise_across_devices() -> None:
    # Two per-device replicas must be concatenated pairwise.
    a0, a1 = _buf(1), _buf(2)
    b0, b1 = _buf(2), _buf(1)

    merged = merge_per_device_buffers([a0, a1], [b0, b1])

    assert len(merged) == 2
    assert merged[0].to_numpy().shape == (3, _HIDDEN)
    assert merged[1].to_numpy().shape == (3, _HIDDEN)


def test_merge_returns_other_side_when_one_is_empty() -> None:
    a, empty = _buf(2), _buf(0)

    # Empty right side -> left returned untouched.
    assert merge_per_device_buffers([a], [empty]) == [a]
    # Empty left side -> right returned untouched.
    assert merge_per_device_buffers([empty], [a]) == [a]
    # Both empty -> first returned.
    assert merge_per_device_buffers([empty], [empty]) == [empty]


_VISION_TOKEN_ID = 98


def _two_image_context() -> Gemma4Context:
    """A 2-image context whose first image has already been encoded.

    Layout (``98`` = vision placeholder)::

        idx:  0  1  2  3 |4  5  6  7| 8  9 10 11 |12 13 14 15| 16 17
                          \\- img0 -/             \\- img1 --/

    ``pixel_position_ids`` is the full per-image list (one entry per image,
    fixed at tokenization). Each image's grid is a 1x4 row so pooling with
    ``k=1`` maps patch ``i`` to output bin ``i``.
    """
    # fmt: off
    tokens = np.array(
        [51, 52, 53, 54, 98, 98, 98, 98, 55, 56, 57, 58, 98, 98, 98, 98, 59, 60],
        dtype=np.int64,
    )
    # fmt: on

    def _pixels() -> np.ndarray:
        return np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    # Distinct grid coords per image so a mis-indexed slice is detectable.
    # img0 is a 1x4 row; img1 is a 2x2 block (both yield 4 pooled bins under
    # k=1, but have different (x, y) coords so the selected slice is verifiable).
    pos0 = np.stack([np.arange(4), np.full(4, 0)], axis=1).astype(np.int32)
    pos1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)

    ctx = Gemma4Context(
        max_length=64,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(start_idx=4, end_idx=8, pixel_values=_pixels()),
            ImageMetadata(start_idx=12, end_idx=16, pixel_values=_pixels()),
        ],
        vision_token_ids=[_VISION_TOKEN_ID],
        mm_token_type_ids=np.zeros(len(tokens), dtype=np.int64),
        pixel_position_ids=[pos0, pos1],
    )
    # Advance past img0's end so only img1 remains unencoded (image_idx == 1),
    # exactly as a prior chunked-prefill chunk would leave the context.
    ctx.tokens.skip_processing(8)
    assert ctx.image_idx == 1
    assert len(ctx.next_images) == 1
    return ctx


def test_build_image_inputs_aligns_pos_ids_after_partial_encode() -> None:
    # Regression for MXSERV-196: ``pixel_position_ids`` is the full per-image
    # list while ``next_images`` only covers not-yet-encoded images. Under
    # chunked prefill (image_idx > 0) the two drift, which previously raised
    # "Expected N pixel_position_ids, got M" (M > N) and crash-looped the
    # worker. The slice must realign them and select img1's position IDs.
    ctx = _two_image_context()
    devices: list[Device] = [CPU()]
    ve_cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(
        max_entries=0  # disabled: no hashing required.
    )

    image_inputs = build_image_inputs(
        context_batch=[ctx],
        uncached=[ctx],
        devices=devices,
        pooling_kernel_size=1,
        ve_cache=ve_cache,
        empty_embeddings=create_empty_embeddings(devices, _HIDDEN),
        dtype=DType.float32,
    )

    assert image_inputs is not None
    assert image_inputs.raw is not None
    packed_pos_ids = image_inputs.raw.pixel_position_ids[0].to_numpy()
    # Must be img1's 2x2 grid, not the already-encoded img0's 1x4 row.
    expected = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    np.testing.assert_array_equal(packed_pos_ids, expected)


# ---------------------------------------------------------------------------
# Video scatter index tests
# ---------------------------------------------------------------------------

_PATCH_DIM = 3  # minimal patch dimension for video frames


def _video_ctx(
    tokens: np.ndarray,
    video_token_ranges: list[tuple[int, int]],
    processed_length: int,
    video_hashes: list[int] | None = None,
    request_id: str = "r0",
) -> Gemma4Context:
    """Build a minimal Gemma4Context with video fields populated."""
    # One dummy frame covering the first video range (patches = range width).
    n_patches = sum(e - s for s, e in video_token_ranges)
    ctx = Gemma4Context(
        request_id=RequestID(request_id),
        max_length=128,
        tokens=TokenBuffer(tokens),
        images=[],
        vision_token_ids=[],
        mm_token_type_ids=np.zeros(len(tokens), dtype=np.int64),
        pixel_position_ids=[],
        video_frame_patches=[
            np.zeros((n_patches, _PATCH_DIM), dtype=np.float32)
        ],
        video_frame_pos_ids=[np.zeros((n_patches, 2), dtype=np.int32)],
        video_frame_patch_counts=[n_patches],
        video_frame_soft_token_counts=[n_patches],
        video_token_ranges=video_token_ranges,
        video_hashes=video_hashes or [],
    )
    if processed_length > 0:
        ctx.tokens.skip_processing(processed_length)
    return ctx


def test_build_video_inputs_non_chunked() -> None:
    # Non-chunked (processed_length=0): scatter indices match old arange behaviour.
    # Token layout: [text, text, VID, VID, VID, text] — video at positions 2..4.
    tokens = np.array([1, 2, 99, 99, 99, 3], dtype=np.int64)
    ctx = _video_ctx(tokens, [(2, 5)], processed_length=0)

    result = build_video_inputs(
        context_batch=[ctx],
        devices=[CPU()],
        pooling_kernel_size=1,
        dtype=DType.float32,
    )

    assert result is not None
    indices = result.token_indices_np
    assert indices is not None
    # batch_offset=0, active_len=6, processed=0 → rel=[2,3,4], all valid.
    np.testing.assert_array_equal(indices, np.array([2, 3, 4], dtype=np.int32))
    assert indices.dtype == np.int32


def test_build_video_inputs_chunked_prefill() -> None:
    # Chunked-prefill: a 6-token prompt split into two 3-token chunks.
    # Video placeholder at absolute positions 3, 4, 5.
    oob = np.iinfo(np.int32).min
    tokens = np.array([1, 2, 3, 99, 99, 99], dtype=np.int64)

    # Chunk 1: processed=0, active=3 (tokens 0..2) — video positions 3,4,5 all OOB.
    ctx1 = _video_ctx(tokens, [(3, 6)], processed_length=0)
    ctx1.tokens.chunk(3)
    result1 = build_video_inputs(
        context_batch=[ctx1],
        devices=[CPU()],
        pooling_kernel_size=1,
        dtype=DType.float32,
    )
    assert result1 is not None
    indices1 = result1.token_indices_np
    assert indices1 is not None
    # All three video positions are outside the chunk window → all OOB.
    np.testing.assert_array_equal(
        indices1, np.array([oob, oob, oob], dtype=np.int32)
    )

    # Chunk 2: processed=3, active=3 (tokens 3..5) — video positions 3,4,5 all valid.
    ctx2 = _video_ctx(tokens, [(3, 6)], processed_length=3)
    result2 = build_video_inputs(
        context_batch=[ctx2],
        devices=[CPU()],
        pooling_kernel_size=1,
        dtype=DType.float32,
    )
    assert result2 is not None
    indices2 = result2.token_indices_np
    assert indices2 is not None
    # rel = [3,4,5] - 3 = [0,1,2], all in [0,3) → valid, batch_offset=0.
    np.testing.assert_array_equal(indices2, np.array([0, 1, 2], dtype=np.int32))

    # Total index count matches across chunks (3 video tokens).
    assert len(indices1) == len(indices2) == 3


def test_build_video_inputs_chunked_straddle() -> None:
    # Video range straddles a chunk boundary: positions 2..5 with chunk size 4.
    # Chunk 1 (processed=0, active=4): positions 2,3 in-window; 4,5 OOB.
    oob = np.iinfo(np.int32).min
    tokens = np.array([1, 2, 99, 99, 99, 99, 3], dtype=np.int64)

    ctx = _video_ctx(tokens, [(2, 6)], processed_length=0)
    ctx.tokens.chunk(4)
    result = build_video_inputs(
        context_batch=[ctx],
        devices=[CPU()],
        pooling_kernel_size=1,
        dtype=DType.float32,
    )
    assert result is not None
    indices = result.token_indices_np
    assert indices is not None
    # positions 2,3 → rel 2,3 → valid; positions 4,5 → rel 4,5 ≥ active_len=4 → OOB.
    np.testing.assert_array_equal(
        indices, np.array([2, 3, oob, oob], dtype=np.int32)
    )
    assert indices.dtype == np.int32


# ---------------------------------------------------------------------------
# Video vision-encoder cache tests
# ---------------------------------------------------------------------------

_VIDEO_HIDDEN = 4


def _video_embed_buf(n_tokens: int) -> list[Buffer]:
    """Single-device fake embedding for a video with n_tokens output tokens."""
    data = np.ones((n_tokens, _VIDEO_HIDDEN), dtype=np.float32)
    return [Buffer.from_numpy(data).to(CPU())]


def _cached_video_ctx(
    video_hash: int,
    n_tokens: int,
    request_id: str = "r0",
) -> Gemma4Context:
    """Build a minimal context with one video whose hash is pre-set."""
    tokens = np.array([1] + [99] * n_tokens + [2], dtype=np.int64)
    return _video_ctx(
        tokens,
        [(1, 1 + n_tokens)],
        processed_length=0,
        video_hashes=[video_hash],
        request_id=request_id,
    )


def test_video_cache_miss_encodes_and_carries_metadata() -> None:
    # A cache miss: build_video_inputs should return a raw path with cache
    # metadata so execute() can call _cache_and_split afterward.
    cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(max_entries=4)
    ctx = _cached_video_ctx(video_hash=0xDEAD_BEEF, n_tokens=4)
    result = build_video_inputs(
        context_batch=[ctx],
        devices=[CPU()],
        pooling_kernel_size=1,
        dtype=DType.float32,
        ve_cache=cache,
        empty_embeddings=create_empty_embeddings([CPU()], _VIDEO_HIDDEN),
    )
    assert result is not None
    assert result.raw is not None  # miss → needs encoding
    assert result.cached_embeddings is None
    assert result.cache_hashes == [0xDEAD_BEEF]
    assert result.cache_per_video_token_counts == [4]


def test_video_cache_hit_returns_cached_embeddings() -> None:
    # Pre-populate the cache, then verify build_video_inputs returns the cached path.
    cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(max_entries=4)
    video_hash = 0x1234_5678
    bufs = _video_embed_buf(3)
    cache.insert(video_hash, bufs, num_tokens=3)

    ctx = _cached_video_ctx(video_hash=video_hash, n_tokens=3)
    empty = create_empty_embeddings([CPU()], _VIDEO_HIDDEN)
    result = build_video_inputs(
        context_batch=[ctx],
        devices=[CPU()],
        pooling_kernel_size=1,
        dtype=DType.float32,
        ve_cache=cache,
        empty_embeddings=empty,
    )
    assert result is not None
    assert result.raw is None  # hit → skip encoding
    assert result.cached_embeddings is not None
    assert result.cached_embeddings[0].shape[0] == 3
    # Scatter indices still populated for the hit path.
    assert result.token_indices_np is not None


def test_video_cache_cross_request_reuse() -> None:
    # Two requests with the same video hash: second request gets a cache hit.
    cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(max_entries=4)
    video_hash = 0xABCD_EF01
    bufs = _video_embed_buf(5)
    cache.insert(video_hash, bufs, num_tokens=5)

    req1 = _cached_video_ctx(video_hash=video_hash, n_tokens=5, request_id="r1")
    req2 = _cached_video_ctx(video_hash=video_hash, n_tokens=5, request_id="r2")
    empty = create_empty_embeddings([CPU()], _VIDEO_HIDDEN)

    for ctx in (req1, req2):
        result = build_video_inputs(
            context_batch=[ctx],
            devices=[CPU()],
            pooling_kernel_size=1,
            dtype=DType.float32,
            ve_cache=cache,
            empty_embeddings=empty,
        )
        assert result is not None
        assert result.raw is None, "Both requests should be cache hits"

    # Both requests acquired a ref.
    entry = cache.lookup(video_hash)
    assert entry is not None
    assert entry.ref_count == 2

    cache.release_request(RequestID("r1"))
    assert entry.ref_count == 1
    cache.release_request(RequestID("r2"))
    assert entry.ref_count == 0


def test_video_inflight_not_evicted() -> None:
    # A video with an active ref must survive when the cache is full.
    cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(max_entries=1)
    video_hash = 0xFEED_FACE
    bufs = _video_embed_buf(6)
    cache.insert(video_hash, bufs, num_tokens=6)
    cache.acquire(RequestID("in-flight"), video_hash)

    # Inserting another entry should not evict the ref-held video.
    other_hash = 0x1111_2222
    cache.insert(other_hash, _video_embed_buf(2), num_tokens=2)
    assert cache.lookup(video_hash) is not None  # still alive

    # Releasing the request makes it evictable.
    cache.release_request(RequestID("in-flight"))
    cache.insert(0x3333_4444, _video_embed_buf(2), num_tokens=2)
    assert cache.lookup(video_hash) is None  # evicted after ref dropped
