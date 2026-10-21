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

"""Checks bounded FP32 FA2 attention/storage with nonuniform binary inputs."""

from std.collections import OptionalReg
from std.math import ceildiv, exp
from std.testing import assert_almost_equal, assert_equal, TestSuite

from max.gpu.host import DeviceContext, FuncAttribute
from layout import TileTensor, row_major
from nn.attention.gpu.mha import mha_single_batch
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import TileTensorMHAOperand, MHAOperand
from nn.attention.mha_utils import FlashAttentionAlgorithm, MHAConfig


def _fa2_kernel[
    dtype: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    config: MHAConfig,
    queries: Int,
    keys: Int,
    query_rows: Int,
](
    q: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    comptime assert dtype == .float32
    # Keep platform-width Int values inside the GPU, not in its launch ABI.
    mha_single_batch[config=config](
        q,
        k,
        v,
        output,
        Float32(0.6931471805599453),
        queries,
        query_rows,
        UInt32(keys - queries),
        keys,
        keys,
        NullMask(),
        0,
        OptionalReg[ImmutTileTensor1D[dtype]](),
    )


def _check_fp32[
    depth: Int, queries: Int, prefix: Int = 8
](ctx: DeviceContext) raises:
    comptime dtype = DType.float32
    comptime keys = 129
    comptime heads = 3
    comptime config = MHAConfig[dtype](
        heads,
        depth,
        num_queries_per_block=32,
        num_keys_per_block=depth,
        BK=32,
        WM=16,
        WN=32,
        num_pipeline_stages=2,
        algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
    )
    comptime query_rows = ceildiv(queries, config.block_m()) * config.block_m()
    comptime key_rows = ceildiv(keys, config.block_n()) * config.block_n()
    comptime query_size = query_rows * heads * depth
    comptime key_size = key_rows * heads * depth
    comptime suffix = 32
    comptime output_size = prefix + query_size + suffix
    comptime assert prefix % 2 == 0, "Legacy SIMD2 output requires 8B alignment"
    comptime assert queries > 0 and queries <= keys
    comptime assert config.block_m() == config.num_warps_m() * config.warp_m()
    comptime assert config.num_warps_m() == 2
    comptime assert heads * depth <= 2147483647 // (config.block_m() + 1)
    comptime assert depth == config.num_warps_n() * config.warp_n()
    comptime scale = Float32(0.6931471805599453)

    var q_host = ctx.enqueue_create_host_buffer[dtype](query_size)
    var k_host = ctx.enqueue_create_host_buffer[dtype](key_size)
    var v_host = ctx.enqueue_create_host_buffer[dtype](key_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var q_after = ctx.enqueue_create_host_buffer[dtype](query_size)
    var k_after = ctx.enqueue_create_host_buffer[dtype](key_size)
    var v_after = ctx.enqueue_create_host_buffer[dtype](key_size)
    var scores = ctx.enqueue_create_host_buffer[.float64](keys)

    for row in range(query_rows):
        for head in range(heads):
            for col in range(depth):
                var idx = (row * heads + head) * depth + col
                var feature = (row + 3 * head) % 7
                var inverted = (row // 7 + head) % 2
                var query_value = Float32(0)
                if col == 0:
                    query_value = Float32(inverted)
                elif col < 8:
                    if col - 1 == feature:
                        query_value = Float32(1 if inverted == 0 else -1)
                else:
                    var pair = (col - 8) // 2
                    query_value = (
                        Float32((3 * row + 5 * head + pair) % 64 - 32) / 32
                    )
                q_host[idx] = query_value if row < queries else Scalar[dtype](
                    3 + Float32(head + col % 3) / 8
                )
    for row in range(key_rows):
        for head in range(heads):
            for col in range(depth):
                var idx = (row * heads + head) * depth + col
                var key_value: Float32
                if col == 0:
                    key_value = Float32(12 if row == 128 else 1)
                elif col < 8:
                    key_value = Float32(
                        6 if row == 128 else (row >> (col - 1)) & 1
                    )
                else:
                    var pair = (col - 8) // 2
                    key_value = (
                        Float32((11 * row + 7 * head + pair) % 64 - 32) / 32
                    )
                    if col % 2 == 1:
                        key_value = -key_value
                k_host[idx] = key_value if row < keys else Scalar[dtype](
                    -4 - Float32(head + col % 5) / 8
                )
                v_host[idx] = Scalar[dtype](
                    Float32(head + 1) / 2
                    + Float32(col % 7) / 32
                    + Float32((row * 7 + head * 3 + col * 5) % 23 - 11) / 16
                ) if row < keys else Scalar[dtype](
                    11 + Float32(head + col % 7) / 4
                )
    for i in range(output_size):
        out_host[i] = Scalar[dtype](42)

    var q_buffer = ctx.enqueue_create_buffer[dtype](query_size)
    var k_buffer = ctx.enqueue_create_buffer[dtype](key_size)
    var v_buffer = ctx.enqueue_create_buffer[dtype](key_size)
    var out_buffer = ctx.enqueue_create_buffer[dtype](output_size)
    assert_equal(Int(out_buffer.unsafe_ptr().unsafe_offset(prefix)) % 8, 0)
    ctx.enqueue_copy(q_buffer, q_host)
    ctx.enqueue_copy(k_buffer, k_host)
    ctx.enqueue_copy(v_buffer, v_host)
    ctx.enqueue_copy(out_buffer, out_host)

    var q = TileTensor(
        q_buffer.unsafe_ptr(), row_major[1, query_rows, heads, depth]()
    ).as_imm()
    var k = TileTensorMHAOperand(
        TileTensor(
            k_buffer.unsafe_ptr(), row_major[1, key_rows, heads, depth]()
        ).as_imm()
    )
    var v = TileTensorMHAOperand(
        TileTensor(
            v_buffer.unsafe_ptr(), row_major[1, key_rows, heads, depth]()
        ).as_imm()
    )

    # Pin the nonpipelined FA2 entry instead of the public dispatcher.
    comptime kernel = _fa2_kernel[
        dtype, type_of(k), type_of(v), config, queries, keys, query_rows
    ]
    comptime smem = config.shared_mem_bytes()
    ctx.enqueue_function[kernel](
        q.unsafe_ptr().as_unsafe_any_origin(),
        k,
        v,
        out_buffer.unsafe_ptr().unsafe_offset(prefix).as_unsafe_any_origin(),
        grid_dim=(ceildiv(queries, config.block_m()), heads, 1),
        block_dim=config.num_threads(),
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
    )
    ctx.enqueue_copy(out_host, out_buffer)
    ctx.enqueue_copy(q_after, q_buffer)
    ctx.enqueue_copy(k_after, k_buffer)
    ctx.enqueue_copy(v_after, v_buffer)
    ctx.synchronize()

    for i in range(query_size):
        assert_equal(q_after[i], q_host[i])
    for i in range(key_size):
        assert_equal(k_after[i], k_host[i])
        assert_equal(v_after[i], v_host[i])

    for row in range(queries):
        for head in range(heads):
            var max_score = Float64(-1e30)
            for key in range(keys):
                var dot = Float64(0)
                for col in range(depth):
                    dot += Float64(
                        q_host[(row * heads + head) * depth + col]
                    ) * Float64(k_host[(key * heads + head) * depth + col])
                scores[key] = dot * Float64(scale)
                max_score = max(max_score, scores[key])
            var denominator = Float64(0)
            for key in range(keys):
                scores[key] = exp(scores[key] - max_score)
                denominator += scores[key]
            for col in range(depth):
                var expected = Float64(0)
                for key in range(keys):
                    expected += scores[key] * Float64(
                        v_host[(key * heads + head) * depth + col]
                    )
                var actual = Float64(
                    out_host[prefix + (row * heads + head) * depth + col]
                )
                assert_almost_equal(
                    actual, expected / denominator, atol=2e-4, rtol=2e-4
                )
    for i in range(prefix):
        assert_equal(out_host[i], Scalar[dtype](42))
    for i in range(prefix + queries * heads * depth, output_size):
        assert_equal(out_host[i], Scalar[dtype](42))


def test_fp32_depth64_queries16_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 16, 8](ctx)


def test_fp32_depth64_queries17_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 17, 8](ctx)


def test_fp32_depth64_queries32_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 32, 8](ctx)


def test_fp32_depth64_queries33_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 33, 8](ctx)


def test_fp32_depth64_queries47_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 47, 8](ctx)


def test_fp32_depth64_queries48_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 48, 8](ctx)


def test_fp32_depth64_queries64_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[64, 64, 8](ctx)


def test_fp32_depth128_queries32_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[128, 32, 8](ctx)


def test_fp32_depth128_queries33_prefix8() raises:
    with DeviceContext() as ctx:
        _check_fp32[128, 33, 8](ctx)


def test_fp32_depth128_queries48_prefix32() raises:
    with DeviceContext() as ctx:
        _check_fp32[128, 48, 32](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
