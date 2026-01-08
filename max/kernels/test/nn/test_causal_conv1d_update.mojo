# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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

from math import exp
from sys.info import simd_width_of

from algorithm.functional import _get_start_indices_of_nth_subvolume
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from layout._fillers import random
from layout.int_tuple import fill_like
from memory import alloc
from nn.causal_conv1d import (
    causal_conv1d_update_cpu,
    causal_conv1d_update_cpu_no_bias,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


@always_inline
fn silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[DType.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Scalar[DType.float32](1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


fn run_causal_conv1d_update[
    dtype: DType,
    has_bias: Bool,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test causal conv1d update kernel against reference implementation."""
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    
    # Input x: (B, C, L)
    var input_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var input_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        input_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    
    # Conv state: (B, C, S)
    var conv_state_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_state_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len))
    )
    
    # Weight: (C, W)
    var weight_heap = alloc[Scalar[dtype]](dim * width)
    var weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )
    
    # Bias: (C,)
    var bias_heap = alloc[Scalar[dtype]](dim)
    var bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    
    # Output: (B, C, L)
    var result_fused_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var result_fused_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        result_fused_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    ).fill(0)
    
    var result_unfused_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var result_unfused_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        result_unfused_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    ).fill(0)
    
    # Copy of conv_state for reference implementation
    var conv_state_ref_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_state_ref_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_ref_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len))
    )
    
    # Initialize input data
    random(input_h)
    random(conv_state_h)
    random(weight_h)
    random(bias_h)
    
    # Copy conv_state for reference
    for i in range(batch * dim * state_len):
        conv_state_ref_h.ptr[i] = conv_state_h.ptr[i]
    
    var input_buf = input_h
    var conv_state_buf = conv_state_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_fused_buf = result_fused_h
    var result_unfused_buf = result_unfused_h
    var conv_state_ref_buf = conv_state_ref_h
    
    # Strides for channel-first layout (B, C, L)
    var x_batch_stride: UInt32 = dim * seqlen
    var x_c_stride: UInt32 = seqlen
    var x_l_stride: UInt32 = 1
    
    var conv_state_batch_stride: UInt32 = dim * state_len
    var conv_state_c_stride: UInt32 = state_len
    var conv_state_l_stride: UInt32 = 1
    
    var weight_c_stride: UInt32 = width
    var weight_width_stride: UInt32 = 1
    
    var out_batch_stride: UInt32 = dim * seqlen
    var out_c_stride: UInt32 = seqlen
    var out_l_stride: UInt32 = 1
    
    var silu_activation = activation == "silu"
    
    # Test fused kernel
    if has_bias:
        causal_conv1d_update_cpu[
            input_buf.dtype,
            input_buf.layout,
            conv_state_buf.dtype,
            conv_state_buf.layout,
            weight_buf.dtype,
            weight_buf.layout,
            result_fused_buf.dtype,
            result_fused_buf.layout,
            bias_buf.dtype,
            bias_buf.layout,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_buf,
            conv_state_buf,
            weight_buf,
            result_fused_buf,
            bias_buf,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            conv_state_batch_stride,
            conv_state_c_stride,
            conv_state_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation,
        )
    else:
        causal_conv1d_update_cpu_no_bias[
            input_buf.dtype,
            input_buf.layout,
            conv_state_buf.dtype,
            conv_state_buf.layout,
            weight_buf.dtype,
            weight_buf.layout,
            result_fused_buf.dtype,
            result_fused_buf.layout,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_buf,
            conv_state_buf,
            weight_buf,
            result_fused_buf,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            conv_state_batch_stride,
            conv_state_c_stride,
            conv_state_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation,
        )
    
    # Reference implementation
    var width_minus_1: Int = width - 1
    
    for b in range(batch):
        for c in range(dim):
            var cur_bias: Scalar[dtype] = Scalar[dtype](0.0)
            if has_bias:
                cur_bias = bias_buf.ptr.offset(c).load()
            
            # Process each position in the input sequence
            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = cur_bias
                
                for w in range(width):
                    # Position in the virtual concatenated sequence [conv_state, x]
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[dtype] = Scalar[dtype](0.0)
                    
                    if src_pos >= state_len:
                        # Read from x
                        var x_l_pos = src_pos - state_len
                        var x_offset = b * x_batch_stride + c * x_c_stride + x_l_pos * x_l_stride
                        input_val = input_buf.ptr.offset(x_offset).load()
                    elif src_pos >= 0:
                        # Read from conv_state
                        var conv_state_offset = b * conv_state_batch_stride + c * conv_state_c_stride + src_pos * conv_state_l_stride
                        input_val = conv_state_ref_buf.ptr.offset(conv_state_offset).load()
                    # else: src_pos < 0, treat as 0 (zero padding)
                    
                    var weight_offset = c * weight_c_stride + w * weight_width_stride
                    var weight_val = weight_buf.ptr.offset(weight_offset).load()
                    conv_sum = conv_sum + input_val * weight_val
                
                # Write output
                var out_offset = b * out_batch_stride + c * out_c_stride + l * out_l_stride
                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)
                result_unfused_buf.ptr.offset(out_offset).store(out_val)
            
            # Update conv_state: shift old values and add new x values
            if seqlen >= state_len:
                # x is longer than state, just copy last state_len values from x
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_offset = b * x_batch_stride + c * x_c_stride + x_l_pos * x_l_stride
                    var x_val = input_buf.ptr.offset(x_offset).load()
                    var conv_state_offset = b * conv_state_batch_stride + c * conv_state_c_stride + s * conv_state_l_stride
                    conv_state_ref_buf.ptr.offset(conv_state_offset).store(x_val)
            else:
                # Shift conv_state left by seqlen positions, then append x
                for s in range(state_len - seqlen):
                    var src_offset = b * conv_state_batch_stride + c * conv_state_c_stride + (s + seqlen) * conv_state_l_stride
                    var dst_offset = b * conv_state_batch_stride + c * conv_state_c_stride + s * conv_state_l_stride
                    var val = conv_state_ref_buf.ptr.offset(src_offset).load()
                    conv_state_ref_buf.ptr.offset(dst_offset).store(val)
                
                # Copy x values to the end
                for l in range(seqlen):
                    var x_offset = b * x_batch_stride + c * x_c_stride + l * x_l_stride
                    var x_val = input_buf.ptr.offset(x_offset).load()
                    var conv_state_offset = b * conv_state_batch_stride + c * conv_state_c_stride + (state_len - seqlen + l) * conv_state_l_stride
                    conv_state_ref_buf.ptr.offset(conv_state_offset).store(x_val)
    
    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.ptr[i],
            result_unfused_h.ptr[i],
            rtol=rtol,
        )
    
    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_h.ptr[i],
            conv_state_ref_h.ptr[i],
            rtol=rtol,
        )
    
    # Cleanup
    input_heap.free()
    conv_state_heap.free()
    conv_state_ref_heap.free()
    weight_heap.free()
    bias_heap.free()
    result_fused_heap.free()
    result_unfused_heap.free()


def main():
    # Test basic cases
    run_causal_conv1d_update[DType.float32, True, "none"](2, 4, 1, 3, 4)
    print("✓ Basic causal conv1d update test passed")
    
    run_causal_conv1d_update[DType.float32, True, "silu"](2, 4, 1, 3, 4)
    print("✓ Causal conv1d update with SiLU test passed")
    
    run_causal_conv1d_update[DType.float32, False, "none"](2, 4, 1, 3, 4)
    print("✓ Causal conv1d update without bias test passed")
    
    # Test with seqlen > 1
    run_causal_conv1d_update[DType.float32, True, "none"](2, 4, 4, 3, 4)
    print("✓ Causal conv1d update with seqlen > 1 test passed")
    
    # Test various widths
    run_causal_conv1d_update[DType.float32, True, "none"](2, 4, 1, 2, 3)
    run_causal_conv1d_update[DType.float32, True, "none"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[DType.float32, True, "none"](2, 4, 1, 4, 5)
    print("✓ Causal conv1d update various widths test passed")
    
    # Test with larger state
    run_causal_conv1d_update[DType.float32, True, "none"](2, 8, 1, 3, 8)
    print("✓ Causal conv1d update with larger state test passed")
    
    # Test combinations
    run_causal_conv1d_update[DType.float32, False, "none"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[DType.float32, True, "none"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[DType.float32, False, "silu"](2, 4, 1, 3, 4)
    run_causal_conv1d_update[DType.float32, True, "silu"](2, 4, 1, 3, 4)
    print("✓ Causal conv1d update combinations test passed")

