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

from layout import MixedLayout, TileTensor, row_major
from state_space.causal_conv1d import (
    causal_conv1d_channel_last_fwd_cpu,
    causal_conv1d_channel_last_fwd_cpu_with_seq_idx,
)
from std.testing import TestSuite, assert_almost_equal


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def run_channel_last_padded[
    with_seq_idx: Bool
](batch: Int, dim: Int, seqlen: Int, width: Int) raises:
    """Channel-last kernels on padded (strided) x, weight and output tensors.

    The tensors are windows into larger buffers, so a kernel that assumed
    contiguous offsets instead of the layout strides reads the wrong data.
    """
    comptime dtype = DType.float32
    var c_pad = dim + 3
    var w_pad = width + 2

    # Logical (B, L, C) views with the channel dimension padded.
    var x_heap = List(length=batch * seqlen * c_pad, fill=Scalar[dtype](-77))
    var out_heap = List(length=batch * seqlen * c_pad, fill=Scalar[dtype](-77))
    # Logical (C, W) view with the width dimension padded.
    var w_heap = List(length=dim * w_pad, fill=Scalar[dtype](-77))
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var seq_heap = List(length=batch * seqlen, fill=Int32(0))

    var x_tt = TileTensor(
        x_heap,
        MixedLayout((batch, seqlen, dim), (seqlen * c_pad, c_pad, 1)),
    )
    var out_tt = TileTensor(
        out_heap,
        MixedLayout((batch, seqlen, dim), (seqlen * c_pad, c_pad, 1)),
    )
    var w_tt = TileTensor(w_heap, MixedLayout((dim, width), (w_pad, 1)))
    var bias_tt = TileTensor(bias_heap, row_major(dim))
    var seq_tt = TileTensor(seq_heap, row_major(batch, seqlen))

    # Deterministic, distinct values.
    for b in range(batch):
        for l in range(seqlen):
            seq_heap[b * seqlen + l] = Int32(l // 3)
            for c in range(dim):
                x_heap[(b * seqlen + l) * c_pad + c] = (
                    Scalar[dtype](((b * 7 + l * 3 + c * 5) % 11) - 5) * 0.25
                )
    for c in range(dim):
        bias_heap[c] = Scalar[dtype](c % 3) * 0.5
        for w in range(width):
            w_heap[c * w_pad + w] = Scalar[dtype](((c * 3 + w * 2) % 5) - 2)

    comptime if with_seq_idx:
        causal_conv1d_channel_last_fwd_cpu_with_seq_idx[
            dtype, dtype, dtype, dtype, DType.int32
        ](
            batch,
            dim,
            seqlen,
            width,
            x_tt.as_imm(),
            w_tt.as_imm(),
            out_tt,
            bias_tt.as_imm(),
            seq_tt.as_imm(),
            False,
        )
    else:
        causal_conv1d_channel_last_fwd_cpu[dtype, dtype, dtype, dtype](
            batch,
            dim,
            seqlen,
            width,
            x_tt.as_imm(),
            w_tt.as_imm(),
            out_tt,
            bias_tt.as_imm(),
            False,
        )

    for b in range(batch):
        for l in range(seqlen):
            for c in range(dim):
                var expected = bias_heap[c]
                for w in range(width):
                    var in_l = l - (width - 1 - w)
                    if in_l < 0:
                        continue
                    comptime if with_seq_idx:
                        if (
                            seq_heap[b * seqlen + in_l]
                            != seq_heap[b * seqlen + l]
                        ):
                            continue
                    expected += (
                        x_heap[(b * seqlen + in_l) * c_pad + c]
                        * w_heap[c * w_pad + w]
                    )
                assert_almost_equal(
                    out_heap[(b * seqlen + l) * c_pad + c], expected
                )
            # Padding channels must stay untouched.
            for c in range(dim, c_pad):
                assert_almost_equal(
                    out_heap[(b * seqlen + l) * c_pad + c], Scalar[dtype](-77)
                )


def test_channel_last_padded() raises:
    """Test channel-last CPU kernel on padded tensors."""
    run_channel_last_padded[False](2, 5, 9, 3)
    run_channel_last_padded[False](1, 4, 8, 4)


def test_channel_last_with_seq_idx_padded() raises:
    """Test channel-last CPU kernel with seq_idx on padded tensors."""
    run_channel_last_padded[True](2, 5, 9, 3)
    run_channel_last_padded[True](1, 4, 8, 4)
