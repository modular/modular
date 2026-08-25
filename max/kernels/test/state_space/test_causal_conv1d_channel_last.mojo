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
"""Tests for the stride-driven causal conv1d CPU core in channel-last layout.

Exercises `causal_conv1d_fwd_cpu` with channel-last (B, L, C) strides, covering
widths 1-4, SiLU on/off, and packed-sequence masking via `seq_idx`.
"""

from std.math import exp

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from state_space.causal_conv1d import causal_conv1d_fwd_cpu
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


@always_inline
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[DType.float32]()
    var sigmoid_x = Scalar[DType.float32](1.0) / (
        Scalar[DType.float32](1.0) + exp(-x_f32)
    )
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_causal_conv1d_channel_last[
    dtype: DType,
    has_seq_idx: Bool,
    activation: StaticString,
](batch: Int, dim: Int, seqlen: Int, width: Int, rtol: Float64 = 0.01) raises:
    """Compare the channel-last CPU core against an explicit naive reference."""
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # Channel-last input/output: (B, L, C).
    var input_heap = List(length=batch * seqlen * dim, fill=Scalar[dtype](0))
    var input_h = LayoutTensor[dtype, layout_3d, _](
        input_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, dim)),
    )
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_h = LayoutTensor[dtype, layout_2d, _](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_h = LayoutTensor[dtype, layout_1d, _](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    var result_heap = List(length=batch * seqlen * dim, fill=Scalar[dtype](0))
    var ref_heap = List(length=batch * seqlen * dim, fill=Scalar[dtype](0))

    random(input_h)
    random(weight_h)
    random(bias_h)

    # Packed-sequence tags (B, L): first half is segment 0, second half is
    # segment 1. Only read when has_seq_idx is True.
    var seq_idx_heap = List(length=batch * seqlen, fill=Scalar[DType.int32](0))
    var seq_idx_h = LayoutTensor[DType.int32, layout_2d, _](
        seq_idx_heap, RuntimeLayout[layout_2d].row_major(Index(batch, seqlen))
    )
    for b in range(batch):
        for l in range(seqlen):
            seq_idx_h.ptr[b * seqlen + l] = Scalar[DType.int32](
                0 if l < seqlen // 2 else 1
            )

    var input_tt = TileTensor(input_heap, row_major(batch, seqlen, dim))
    var weight_tt = TileTensor(weight_heap, row_major(dim, width))
    var bias_tt = TileTensor(bias_heap, row_major(dim))
    var result_tt = TileTensor(result_heap, row_major(batch, seqlen, dim))
    var seq_idx_tt = TileTensor(seq_idx_heap, row_major(batch, seqlen))

    # Channel-last strides for (B, L, C).
    var x_batch_stride: UInt32 = UInt32(seqlen * dim)
    var x_l_stride: UInt32 = UInt32(dim)
    var x_c_stride: UInt32 = 1
    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var bias_stride: UInt32 = 1
    var seq_idx_batch_stride: UInt32 = UInt32(seqlen)
    var seq_idx_l_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    causal_conv1d_fwd_cpu[
        dtype, dtype, dtype, dtype, DType.int32, True, has_seq_idx
    ](
        batch,
        dim,
        seqlen,
        width,
        input_tt.as_immut(),
        weight_tt.as_immut(),
        result_tt,
        bias_tt.as_immut(),
        seq_idx_tt.as_immut(),
        x_batch_stride,
        x_c_stride,
        x_l_stride,
        weight_c_stride,
        weight_width_stride,
        x_batch_stride,
        x_c_stride,
        x_l_stride,
        bias_stride,
        seq_idx_batch_stride,
        seq_idx_l_stride,
        silu_activation,
    )

    # Naive channel-last reference with optional packed-sequence masking.
    var width_minus_1 = width - 1
    for b in range(batch):
        for l in range(seqlen):
            var cur_seq = seq_idx_heap[b * seqlen + l]
            for c in range(dim):
                var conv_sum: Scalar[dtype] = bias_heap[c]
                for w in range(width):
                    var input_l = l - (width_minus_1 - w)
                    if input_l < 0:
                        continue

                    comptime if has_seq_idx:
                        if seq_idx_heap[b * seqlen + input_l] != cur_seq:
                            continue
                    var x_val = input_heap[b * seqlen * dim + input_l * dim + c]
                    var w_val = weight_heap[c * width + w]
                    conv_sum = conv_sum + x_val * w_val
                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)
                ref_heap[b * seqlen * dim + l * dim + c] = out_val

    for i in range(batch * seqlen * dim):
        assert_almost_equal(result_heap[i], ref_heap[i], rtol=rtol)


def test_channel_last_basic() raises:
    run_causal_conv1d_channel_last[DType.float32, False, "none"](2, 4, 8, 3)


def test_channel_last_silu() raises:
    run_causal_conv1d_channel_last[DType.float32, False, "silu"](2, 4, 8, 3)


def test_channel_last_widths() raises:
    run_causal_conv1d_channel_last[DType.float32, False, "none"](2, 8, 16, 1)
    run_causal_conv1d_channel_last[DType.float32, False, "none"](2, 8, 16, 2)
    run_causal_conv1d_channel_last[DType.float32, False, "none"](2, 8, 16, 3)
    run_causal_conv1d_channel_last[DType.float32, False, "none"](2, 8, 16, 4)


def test_channel_last_silu_width_4() raises:
    run_causal_conv1d_channel_last[DType.float32, False, "silu"](2, 8, 16, 4)


def test_channel_last_large_sequence() raises:
    run_causal_conv1d_channel_last[DType.float32, False, "none"](2, 16, 128, 4)


def test_channel_last_seq_idx() raises:
    """Packed-sequence masking must not read across the segment boundary."""
    run_causal_conv1d_channel_last[DType.float32, True, "none"](2, 8, 16, 4)


def test_channel_last_seq_idx_silu() raises:
    run_causal_conv1d_channel_last[DType.float32, True, "silu"](2, 8, 16, 3)
