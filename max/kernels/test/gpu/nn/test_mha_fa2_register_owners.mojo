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

"""Checks the FA2 register path against a host attention reference."""

from std.collections import OptionalReg
from std.math import ceildiv, exp
from std.testing import assert_almost_equal, assert_equal

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
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    # Keep platform-width Int values inside the GPU, not in its launch ABI.
    mha_single_batch[config=config](
        q,
        k,
        v,
        output,
        Float32(0.125),
        queries,
        query_rows,
        UInt32(keys - queries),
        keys,
        keys,
        NullMask(),
        0,
        OptionalReg[ImmutTileTensor1D[dtype]](),
    )


def _check_fa2[
    dtype: DType, queries: Int, keys: Int
](ctx: DeviceContext,) raises:
    comptime heads = 2
    comptime depth = 64
    comptime config = MHAConfig[dtype](
        heads,
        depth,
        num_queries_per_block=32,
        num_keys_per_block=64,
        BK=32,
        WM=16,
        WN=64,
        num_pipeline_stages=2,
        algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
    )
    comptime query_rows = ceildiv(queries, config.block_m()) * config.block_m()
    comptime key_rows = ceildiv(keys, config.block_n()) * config.block_n()
    comptime query_size = query_rows * heads * depth
    comptime key_size = key_rows * heads * depth
    comptime guard = 32
    comptime scale = Float32(0.125)

    var q_host = ctx.enqueue_create_host_buffer[dtype](query_size)
    var k_host = ctx.enqueue_create_host_buffer[dtype](key_size)
    var v_host = ctx.enqueue_create_host_buffer[dtype](key_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](query_size + guard)
    var scores = ctx.enqueue_create_host_buffer[.float64](keys)

    for row in range(query_rows):
        for head in range(heads):
            for col in range(depth):
                var idx = (row * heads + head) * depth + col
                q_host[idx] = Scalar[dtype](
                    Float32((row * 3 + head * 5 + col) % 17 - 8) / 16
                ) if row < queries else Scalar[dtype](0)
    for row in range(key_rows):
        for head in range(heads):
            for col in range(depth):
                var idx = (row * heads + head) * depth + col
                k_host[idx] = Scalar[dtype](
                    Float32((row * 11 + head * 5 + col) % 17 - 8) / 16
                ) if row < keys else Scalar[dtype](0)
                v_host[idx] = Scalar[dtype](
                    Float32(head + 1) / 2
                    + Float32(col % 7) / 32
                    + Float32((row * 7 + head * 3 + col * 5) % 23 - 11) / 16
                ) if row < keys else Scalar[dtype](0)
    for i in range(query_size + guard):
        out_host[i] = Scalar[dtype](42)

    var q_buffer = ctx.enqueue_create_buffer[dtype](query_size)
    var k_buffer = ctx.enqueue_create_buffer[dtype](key_size)
    var v_buffer = ctx.enqueue_create_buffer[dtype](key_size)
    var out_buffer = ctx.enqueue_create_buffer[dtype](query_size + guard)
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

    # Algorithm selection alone still routes B200 through FA4; call FA2 directly.
    comptime kernel = _fa2_kernel[
        dtype, type_of(k), type_of(v), config, queries, keys, query_rows
    ]
    comptime smem = config.shared_mem_bytes()
    ctx.enqueue_function[kernel](
        q.unsafe_ptr().as_unsafe_any_origin(),
        k,
        v,
        out_buffer.unsafe_ptr().as_unsafe_any_origin(),
        grid_dim=(ceildiv(queries, config.block_m()), heads, 1),
        block_dim=config.num_threads(),
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
    )
    ctx.enqueue_copy(out_host, out_buffer)
    ctx.synchronize()

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
                    out_host[(row * heads + head) * depth + col]
                )
                assert_almost_equal(
                    actual, expected / denominator, atol=2e-3, rtol=1e-2
                )
    for i in range(queries * heads * depth, query_size + guard):
        assert_equal(out_host[i], Scalar[dtype](42))


def main() raises:
    with DeviceContext() as ctx:
        _check_fa2[.bfloat16, 32, 128](ctx)
        _check_fa2[.float16, 33, 129](ctx)
