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

"""Checks the generic MLA prefill kernel against a host attention reference."""

from std.math import ceildiv, exp
from std.testing import assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext, FuncAttribute
from layout import TileTensor, row_major
from nn.attention.gpu.mla import mla_prefill_single_batch
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import TileTensorMHAOperand, MHAOperand
from nn.attention.mha_utils import FlashAttentionAlgorithm, MHAConfig


def _mla_prefill_kernel[
    dtype: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    config: MHAConfig,
    queries: Int,
    keys: Int,
    query_rows: Int,
](
    q: MutPointer[Scalar[dtype], MutAnyOrigin],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    # Keep platform-width Int values inside the GPU, not in its launch ABI.
    mla_prefill_single_batch[
        config=config, group=1, q_depth=192, cache_depth=64
    ](
        q,
        k,
        v,
        k_rope,
        output,
        Float32(0.125),
        queries,
        query_rows,
        UInt32(keys - queries),
        UInt32(0),
        keys,
        NullMask(),
        0,
    )


def _check_mla_prefill[
    dtype: DType, queries: Int, keys: Int
](ctx: DeviceContext) raises:
    comptime heads = 2
    comptime depth = 128
    comptime rope = 64
    comptime q_depth = depth + rope
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
    comptime q_size = query_rows * heads * q_depth
    comptime k_size = key_rows * heads * depth
    comptime rope_size = key_rows * heads * rope
    comptime out_size = query_rows * heads * depth
    comptime guard = 32
    comptime scale = Float32(0.125)

    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var k_host = ctx.enqueue_create_host_buffer[dtype](k_size)
    var v_host = ctx.enqueue_create_host_buffer[dtype](k_size)
    var rope_host = ctx.enqueue_create_host_buffer[dtype](rope_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](out_size + guard)
    var scores = ctx.enqueue_create_host_buffer[.float64](keys)

    for row in range(query_rows):
        for head in range(heads):
            for col in range(q_depth):
                q_host[(row * heads + head) * q_depth + col] = Scalar[dtype](
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
            for col in range(rope):
                rope_host[(row * heads + head) * rope + col] = Scalar[dtype](
                    Float32((row * 5 + head * 7 + col * 3) % 13 - 6) / 16
                ) if row < keys else Scalar[dtype](0)
    for i in range(out_size + guard):
        out_host[i] = Scalar[dtype](42)

    var q_buffer = ctx.enqueue_create_buffer[dtype](q_size)
    var k_buffer = ctx.enqueue_create_buffer[dtype](k_size)
    var v_buffer = ctx.enqueue_create_buffer[dtype](k_size)
    var rope_buffer = ctx.enqueue_create_buffer[dtype](rope_size)
    var out_buffer = ctx.enqueue_create_buffer[dtype](out_size + guard)
    ctx.enqueue_copy(q_buffer, q_host)
    ctx.enqueue_copy(k_buffer, k_host)
    ctx.enqueue_copy(v_buffer, v_host)
    ctx.enqueue_copy(rope_buffer, rope_host)
    ctx.enqueue_copy(out_buffer, out_host)

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
    var k_rope = TileTensorMHAOperand(
        TileTensor(
            rope_buffer.unsafe_ptr(), row_major[1, key_rows, heads, rope]()
        ).as_imm()
    )

    comptime kernel = _mla_prefill_kernel[
        dtype,
        type_of(k),
        type_of(v),
        type_of(k_rope),
        config,
        queries,
        keys,
        query_rows,
    ]
    # Q, K, V and P tiles plus the cross-warp softmax scratch.
    comptime smem = (
        (
            config.block_m() * q_depth
            + config.block_n() * q_depth
            + config.block_n() * depth
            + config.block_m() * config.block_n()
        )
        * 2
        + 4096
    )
    ctx.enqueue_function[kernel](
        q_buffer.unsafe_ptr().as_unsafe_any_origin(),
        k,
        v,
        k_rope,
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
                        q_host[(row * heads + head) * q_depth + col]
                    ) * Float64(k_host[(key * heads + head) * depth + col])
                for col in range(rope):
                    dot += Float64(
                        q_host[(row * heads + head) * q_depth + depth + col]
                    ) * Float64(rope_host[(key * heads + head) * rope + col])
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
                assert_almost_equal(
                    Float64(out_host[(row * heads + head) * depth + col]),
                    expected / denominator,
                    atol=3e-3,
                    rtol=2e-2,
                )
    for i in range(out_size, out_size + guard):
        assert_equal(out_host[i], Scalar[dtype](42))


def main() raises:
    with DeviceContext() as ctx:
        _check_mla_prefill[.bfloat16, 32, 128](ctx)
        _check_mla_prefill[.bfloat16, 33, 129](ctx)
