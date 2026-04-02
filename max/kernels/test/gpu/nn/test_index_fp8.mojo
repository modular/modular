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

from std.gpu.host import DeviceContext
from nn.index_fp8 import fp8_index, fp8_index_naive
from std.random import rand
from layout import Idx, TileTensor, row_major
from std.testing import assert_almost_equal


def test_index_fp8[
    num_heads: Int,
    depth: Int,
](batch_size: Int, seq_len: Int, num_keys: Int, ctx: DeviceContext) raises:
    print(
        "test_index_fp8 with params:",
        "num_heads:",
        num_heads,
        "depth:",
        depth,
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
    )
    var q_size = batch_size * seq_len * num_heads * depth
    var qs_size = batch_size * seq_len * num_heads
    var k_size = batch_size * num_keys * depth
    var ks_size = batch_size * num_keys
    var o_size = batch_size * seq_len * num_keys

    var q_ptr = alloc[Scalar[DType.float8_e4m3fn]](q_size)
    var qs_ptr = alloc[Scalar[DType.float32]](qs_size)
    var k_ptr = alloc[Scalar[DType.float8_e4m3fn]](k_size)
    var ks_ptr = alloc[Scalar[DType.float32]](ks_size)
    var o_ptr = alloc[Scalar[DType.float32]](o_size)
    var o_ref_ptr = alloc[Scalar[DType.float32]](o_size)
    var input_row_offsets = alloc[UInt32](batch_size + 1)
    var cache_row_offsets = alloc[UInt32](batch_size + 1)

    var q_device_ptr = ctx.enqueue_create_buffer[DType.float8_e4m3fn](q_size)
    var qs_device_ptr = ctx.enqueue_create_buffer[DType.float32](qs_size)
    var k_device_ptr = ctx.enqueue_create_buffer[DType.float8_e4m3fn](k_size)
    var ks_device_ptr = ctx.enqueue_create_buffer[DType.float32](ks_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var o_device_ptr = ctx.enqueue_create_buffer[DType.float32](o_size)
    var o_device_ref_ptr = ctx.enqueue_create_buffer[DType.float32](o_size)

    rand(q_ptr, q_size)
    rand(qs_ptr, qs_size)
    rand(k_ptr, k_size)
    rand(ks_ptr, ks_size)

    # input row offsets and cache row offsets
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(i * seq_len)
        cache_row_offsets[i] = UInt32(i * num_keys)
    input_row_offsets[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets[batch_size] = UInt32(batch_size * num_keys)

    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(qs_device_ptr, qs_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(ks_device_ptr, ks_ptr)
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    # Ragged Q tensor: [total_seq_len, num_heads, depth]
    var q_device = TileTensor(
        q_device_ptr.unsafe_ptr().as_immutable(),
        row_major((Idx(batch_size * seq_len), Idx[num_heads](), Idx[depth]())),
    )

    var qs_device = TileTensor(
        qs_device_ptr.unsafe_ptr(),
        row_major((Idx(batch_size * seq_len), Idx[num_heads]())),
    )

    var k_device = TileTensor(
        k_device_ptr.unsafe_ptr().as_immutable(),
        row_major((Idx(batch_size * num_keys), Idx[1](), Idx[depth]())),
    )

    var ks_device = TileTensor(
        ks_device_ptr.unsafe_ptr(),
        row_major(Idx(batch_size * num_keys)),
    )

    var o_device = TileTensor(
        o_device_ptr.unsafe_ptr(),
        row_major((Idx(batch_size * seq_len), Idx(num_keys))),
    )

    var o_ref_device = TileTensor(
        o_device_ref_ptr.unsafe_ptr(),
        row_major((Idx(batch_size * seq_len), Idx(num_keys))),
    )

    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    var cache_row_offsets_device = TileTensor(
        cache_row_offsets_device_ptr.unsafe_ptr().as_immutable(),
        row_major(Idx(batch_size + 1)),
    )

    fp8_index[num_heads, depth](
        o_device,
        q_device,
        qs_device,
        k_device,
        ks_device,
        input_row_offsets_device,
        cache_row_offsets_device,
        batch_size,
        seq_len,
        num_keys,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(o_ptr, o_device_ptr)

    fp8_index_naive[num_heads, depth](
        o_ref_device,
        q_device,
        qs_device,
        k_device,
        ks_device,
        input_row_offsets_device,
        cache_row_offsets_device,
        batch_size,
        seq_len,
        num_keys,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(o_ref_ptr, o_device_ref_ptr)

    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(num_keys):
                var expect = o_ref_ptr[
                    b * seq_len * num_keys + s * num_keys + k
                ]
                var actual = o_ptr[b * seq_len * num_keys + s * num_keys + k]

                if abs((actual - expect)) > 1e-2:
                    print(b, s, k, actual, expect)
                assert_almost_equal(actual, expect, atol=1e-2, rtol=1e-3)

    _ = q_device_ptr
    _ = qs_device_ptr
    _ = k_device_ptr
    _ = ks_device_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr
    _ = o_device_ptr
    _ = o_device_ref_ptr

    q_ptr.free()
    qs_ptr.free()
    k_ptr.free()
    ks_ptr.free()
    input_row_offsets.free()
    cache_row_offsets.free()
    o_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        test_index_fp8[num_heads=128, depth=128](2, 128, 128, ctx)
        test_index_fp8[num_heads=128, depth=128](2, 32, 32, ctx)
        test_index_fp8[num_heads=128, depth=128](4, 200, 200, ctx)
        test_index_fp8[num_heads=128, depth=128](1, 501, 501, ctx)
        test_index_fp8[num_heads=128, depth=128](3, 600, 600, ctx)
        test_index_fp8[num_heads=128, depth=128](4, 722, 722, ctx)
        test_index_fp8[num_heads=128, depth=128](5, 32, 64, ctx)
        test_index_fp8[num_heads=128, depth=128](2, 128, 256, ctx)

        test_index_fp8[num_heads=64, depth=128](2, 128, 128, ctx)
        test_index_fp8[num_heads=64, depth=128](2, 32, 32, ctx)
        test_index_fp8[num_heads=64, depth=128](4, 200, 200, ctx)
        test_index_fp8[num_heads=64, depth=128](1, 501, 501, ctx)
        test_index_fp8[num_heads=64, depth=128](3, 600, 600, ctx)
        test_index_fp8[num_heads=64, depth=128](4, 722, 722, ctx)
        test_index_fp8[num_heads=64, depth=128](5, 32, 64, ctx)
        test_index_fp8[num_heads=64, depth=128](2, 128, 256, ctx)
