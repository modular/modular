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
    create_empty_embeddings,
    merge_per_device_buffers,
)
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.context import ImageMetadata
from max.pipelines.context.context import TokenBuffer
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache

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
