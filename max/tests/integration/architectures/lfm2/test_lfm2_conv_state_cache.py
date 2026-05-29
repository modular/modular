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
"""Regression tests for ``ConvStateCache`` buffer concatenation.

Covers the bfloat16 path that previously failed with
``RuntimeError: Unsupported dtype in DLTensor`` because
``Buffer.to_numpy`` round-trips through ``np.from_dlpack``, and numpy has
no native bfloat16 support.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.pipelines.architectures.lfm2.model import (
    _cat_buffers,
    _split_buffer_dim0,
)


def test_cat_buffers_bfloat16_does_not_raise() -> None:
    device = CPU()
    # Build two [1, hidden, kernel] bf16 buffers via the uint16 byte view.
    shape = (1, 4, 3)
    a_u16 = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    b_u16 = (a_u16 + 100).astype(np.uint16)
    a = Buffer.from_numpy(a_u16).to(device).view(DType.bfloat16)
    b = Buffer.from_numpy(b_u16).to(device).view(DType.bfloat16)

    out = _cat_buffers([a, b], device)

    assert out.dtype == DType.bfloat16
    assert tuple(out.shape) == (2, 4, 3)
    # Verify the bytes survived the round-trip by viewing back to uint16.
    out_u16 = out.view(DType.uint16).to_numpy()
    np.testing.assert_array_equal(
        out_u16, np.concatenate([a_u16, b_u16], axis=0)
    )


def test_cat_buffers_float32_unchanged() -> None:
    device = CPU()
    shape = (1, 4, 3)
    a_f32 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b_f32 = (a_f32 + 100.0).astype(np.float32)
    a = Buffer.from_numpy(a_f32).to(device)
    b = Buffer.from_numpy(b_f32).to(device)

    out = _cat_buffers([a, b], device)

    assert out.dtype == DType.float32
    assert tuple(out.shape) == (2, 4, 3)
    np.testing.assert_array_equal(
        out.to_numpy(), np.concatenate([a_f32, b_f32], axis=0)
    )


def test_split_buffer_dim0_bfloat16_does_not_raise() -> None:
    device = CPU()
    # Build a [2, hidden, kernel] bf16 buffer via the uint16 byte view.
    shape = (2, 4, 3)
    src_u16 = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    buf = Buffer.from_numpy(src_u16).to(device).view(DType.bfloat16)

    pieces = _split_buffer_dim0(buf, device)

    assert len(pieces) == 2
    for i, piece in enumerate(pieces):
        assert piece.dtype == DType.bfloat16
        assert tuple(piece.shape) == (1, 4, 3)
        # Verify byte contents survived the round-trip via uint16 view.
        np.testing.assert_array_equal(
            piece.view(DType.uint16).to_numpy(), src_u16[i : i + 1]
        )


def test_split_buffer_dim0_float32_unchanged() -> None:
    device = CPU()
    shape = (2, 4, 3)
    src_f32 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    buf = Buffer.from_numpy(src_f32).to(device)

    pieces = _split_buffer_dim0(buf, device)

    assert len(pieces) == 2
    for i, piece in enumerate(pieces):
        assert piece.dtype == DType.float32
        assert tuple(piece.shape) == (1, 4, 3)
        np.testing.assert_array_equal(piece.to_numpy(), src_f32[i : i + 1])
