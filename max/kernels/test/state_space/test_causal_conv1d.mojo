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

from std.math import exp

from layout import TileTensor, row_major
from layout._fillers import random
from state_space.causal_conv1d import (
    causal_conv1d_channel_first_fwd_cpu,
)
from std.testing import TestSuite, assert_almost_equal


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


@inline(.always)
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Float32(1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_causal_conv1d[
    dtype: DType,
    activation: StaticString,
](batch: Int, dim: Int, seqlen: Int, width: Int, rtol: Float64 = 0.01,) raises:
    """Test causal conv1d kernel against reference implementation."""
    # Allocate host memory
    var input_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var input_h = TileTensor(
        input_heap,
        row_major(batch, dim, seqlen),
    )
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_h = TileTensor(weight_heap, row_major(dim, width))
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_h = TileTensor(bias_heap, row_major(dim))
    var result_fused_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var result_fused_h = TileTensor(
        result_fused_heap,
        row_major(batch, dim, seqlen),
    )
    var result_unfused_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var result_unfused_h = TileTensor(
        result_unfused_heap,
        row_major(batch, dim, seqlen),
    )

    # Initialize input data
    random(input_h)
    random(weight_h)
    random(bias_h)

    var input_buf = input_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_unfused_buf = result_unfused_h

    # Strides for channel-first layout (B, C, L)
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_c_stride: UInt32 = UInt32(seqlen)
    var x_l_stride: UInt32 = 1
    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_c_stride: UInt32 = UInt32(seqlen)
    var out_l_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    # Test kernel
    causal_conv1d_channel_first_fwd_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
    ](
        batch,
        dim,
        seqlen,
        width,
        input_h,
        weight_h,
        result_fused_h,
        bias_h,
        silu_activation,
    )

    # Reference implementation
    var width_minus_1: Int = width - 1
    for b in range(batch):
        for c in range(dim):
            var cur_bias = bias_buf.unsafe_ptr().load(c)
            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = cur_bias
                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    if input_l >= 0:
                        var x_offset = (
                            UInt32(b) * x_batch_stride
                            + UInt32(c) * x_c_stride
                            + UInt32(input_l) * x_l_stride
                        )
                        var input_val = input_buf.unsafe_ptr().load(x_offset)
                        var weight_offset = (
                            UInt32(c) * weight_c_stride
                            + UInt32(w) * weight_width_stride
                        )
                        var weight_val = weight_buf.unsafe_ptr().load(
                            weight_offset
                        )
                        conv_sum = conv_sum + input_val * weight_val

                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)

                var out_offset = (
                    UInt32(b) * out_batch_stride
                    + UInt32(c) * out_c_stride
                    + UInt32(l) * out_l_stride
                )
                result_unfused_buf.unsafe_ptr().store(out_offset, out_val)

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.unsafe_ptr()[i],
            result_unfused_h.unsafe_ptr()[i],
            rtol=rtol,
        )


def test_basic_causal_conv1d() raises:
    """Test basic causal conv1d without activation."""
    run_causal_conv1d[.float32, "none"](2, 4, 8, 3)


def test_causal_conv1d_with_silu() raises:
    """Test causal conv1d with SiLU activation."""
    run_causal_conv1d[.float32, "silu"](2, 4, 8, 3)


def test_causal_conv1d_width_4() raises:
    """Test causal conv1d with kernel width 4."""
    run_causal_conv1d[.float32, "none"](2, 8, 16, 4)


def test_causal_conv1d_silu_width_3() raises:
    """Test causal conv1d with SiLU activation and width 3."""
    run_causal_conv1d[.float32, "silu"](2, 8, 16, 3)


def test_causal_conv1d_various_widths() raises:
    """Test causal conv1d with various kernel widths."""
    run_causal_conv1d[.float32, "none"](2, 4, 8, 1)
    run_causal_conv1d[.float32, "none"](2, 4, 8, 2)
    run_causal_conv1d[.float32, "none"](2, 4, 8, 3)
    run_causal_conv1d[.float32, "none"](2, 4, 8, 4)


def test_causal_conv1d_large_sequence() raises:
    """Test causal conv1d with larger sequence length."""
    run_causal_conv1d[.float32, "none"](2, 16, 128, 3)
