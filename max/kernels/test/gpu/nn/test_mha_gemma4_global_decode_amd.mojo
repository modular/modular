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

"""Correctness test for the Gemma 4 global decode attention shape on AMD.

Regression test for KERN-2826.

Gemma 4's global layers use 32 query heads, 4 KV heads, and head_dim=512.
This compares the gfx950 flash attention decode kernel against the existing
naive GPU reference for a numerically stressful two-tile causal decode shape.
"""

from std.math import isclose

from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from std.gpu.host import DeviceContext
from std.testing import assert_almost_equal
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf


def test_gemma4_global_decode(ctx: DeviceContext) raises:
    print("gemma4 global decode")

    comptime batch_size = 1
    comptime seq_len = 1
    comptime num_keys = 512
    comptime num_heads = 32
    comptime kv_num_heads = 4
    comptime group = num_heads // kv_num_heads
    comptime depth = 512
    comptime dtype = DType.bfloat16
    comptime mask_type = DType.bfloat16
    comptime scale = Float32(1.0)

    var q_size = batch_size * seq_len * num_heads * depth
    var kv_size = batch_size * num_keys * kv_num_heads * depth
    var output_size = q_size
    var mask_size = batch_size * num_heads * seq_len * num_keys

    var q_ptr = ctx.enqueue_create_host_buffer[dtype](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[dtype](kv_size)
    var v_ptr = ctx.enqueue_create_host_buffer[dtype](kv_size)
    var mask_ptr = ctx.enqueue_create_host_buffer[mask_type](mask_size)
    var actual_ptr = ctx.enqueue_create_host_buffer[dtype](output_size)
    var expect_ptr = ctx.enqueue_create_host_buffer[dtype](output_size)

    for i in range(seq_len):
        for h in range(num_heads):
            for d in range(depth):
                q_ptr[(i * num_heads + h) * depth + d] = Scalar[dtype](5.0)
    for i in range(num_keys):
        var k_val = Scalar[dtype](100.0) if i < 128 else Scalar[dtype](1.0)
        var v_val = Scalar[dtype](1.0) if i < 128 else Scalar[dtype](-1.0)
        for h in range(kv_num_heads):
            for d in range(depth):
                k_ptr[(i * kv_num_heads + h) * depth + d] = k_val
                v_ptr[(i * kv_num_heads + h) * depth + d] = v_val

    comptime mask_layout = Layout.row_major[4]()
    var mask = LayoutTensor[mask_type, mask_layout](
        mask_ptr,
        RuntimeLayout[mask_layout].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(seq_len):
                for k_idx in range(num_keys):
                    mask[b, h, q_idx, k_idx] = (
                        0 if q_idx + num_keys - seq_len
                        >= k_idx else min_or_neg_inf[mask_type]()
                    )

    var q_device_ptr = ctx.enqueue_create_buffer[dtype](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[dtype](kv_size)
    var v_device_ptr = ctx.enqueue_create_buffer[dtype](kv_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var actual_device_ptr = ctx.enqueue_create_buffer[dtype](output_size)
    var expect_device_ptr = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    var q_device = TileTensor(
        q_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_device = TileTensor(
        k_device_ptr,
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var v_device = TileTensor(
        v_device_ptr,
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var mask_device = TileTensor(
        mask_device_ptr,
        row_major(
            (Idx(batch_size), Idx(num_heads), Idx(seq_len), Idx(num_keys))
        ),
    )
    var actual_device = TileTensor(
        actual_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var expect_device = TileTensor(
        expect_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )

    flash_attention(
        actual_device,
        q_device,
        k_device,
        v_device,
        CausalMask(),
        scale,
        ctx,
    )

    mha_gpu_naive(
        q_device,
        k_device,
        v_device,
        mask_device,
        expect_device,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(actual_ptr, actual_device_ptr)
    ctx.enqueue_copy(expect_ptr, expect_device_ptr)
    ctx.synchronize()

    for h in range(num_heads):
        for d in range(depth):
            var idx = d + depth * h
            var actual = actual_ptr[idx].cast[DType.float64]()
            var expect = expect_ptr[idx].cast[DType.float64]()
            if not isclose(actual, expect, atol=1e-5, rtol=3e-2):
                print("MISMATCH:", h, d, actual, expect)
            assert_almost_equal(actual, expect, atol=1e-5, rtol=3e-2)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = actual_device_ptr
    _ = expect_device_ptr


def main() raises:
    with DeviceContext() as ctx:
        test_gemma4_global_decode(ctx)
