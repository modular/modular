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

"""Checks non-pipelined decoding register owners against host attention."""

from std.collections import OptionalReg
from std.math import exp
from std.sys import size_of
from std.testing import assert_almost_equal, assert_equal
from std.utils.numerics import get_accum_type

from max.gpu.host import DeviceContext, FuncAttribute
from layout import TileTensor, row_major
from nn.attention.gpu.mha import mha_decoding_single_batch
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import TileTensorMHAOperand, MHAOperand


def _decode_kernel[
    dtype: DType, k_t: MHAOperand, v_t: MHAOperand
](
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    exp_sum: UnsafePointer[Scalar[get_accum_type[dtype]()], MutAnyOrigin],
    qk_max: UnsafePointer[Scalar[get_accum_type[dtype]()], MutAnyOrigin],
):
    # Keep platform-width Int values inside the GPU launch boundary.
    mha_decoding_single_batch[
        BM=16,
        BN=64,
        BK=32,
        WM=16,
        WN=64,
        depth=64,
        num_heads=4,
        num_threads=32,
        num_pipeline_stages=2,
        group=4,
        decoding_warp_split_k=False,
    ](
        q,
        k,
        v,
        output,
        exp_sum,
        qk_max,
        Float32(0.125),
        128,
        1,
        NullMask(),
        0,
        OptionalReg[ImmutTileTensor1D[dtype]](),
    )


def _check_decode[dtype: DType](ctx: DeviceContext) raises:
    comptime heads = 4
    comptime depth = 64
    comptime query_rows = 16
    comptime keys = 128
    comptime query_size = query_rows * depth
    comptime key_size = keys * depth
    comptime guard = 32
    comptime output_size = query_size + 2 * guard
    comptime scale = Float32(0.125)
    comptime accum_type = get_accum_type[dtype]()
    # Q, K, V, P tiles plus the two-row Float32 reduction scratch.
    comptime smem = (16 * 64 + 64 * 64 + 64 * 64 + 16 * 64) * size_of[
        Scalar[dtype]
    ]() + 2 * 16 * size_of[Scalar[accum_type]]()

    var q_host = ctx.enqueue_create_host_buffer[dtype](query_size)
    var k_host = ctx.enqueue_create_host_buffer[dtype](key_size)
    var v_host = ctx.enqueue_create_host_buffer[dtype](key_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var scores = ctx.enqueue_create_host_buffer[.float64](keys)

    for row in range(query_rows):
        for col in range(depth):
            q_host[row * depth + col] = Scalar[dtype](
                Float32((row * 3 + col) % 17 - 8) / 16
            ) if row < heads else Scalar[dtype](0)
    for row in range(keys):
        for col in range(depth):
            k_host[row * depth + col] = Scalar[dtype](
                Float32((row * 11 + col) % 17 - 8) / 16
            )
            v_host[row * depth + col] = Scalar[dtype](
                Float32(0.5)
                + Float32(col % 7) / 32
                + Float32((row * 7 + col * 5) % 23 - 11) / 16
            )
    for i in range(output_size):
        out_host[i] = Scalar[dtype](42)

    var q_buffer = ctx.enqueue_create_buffer[dtype](query_size)
    var k_buffer = ctx.enqueue_create_buffer[dtype](key_size)
    var v_buffer = ctx.enqueue_create_buffer[dtype](key_size)
    var out_buffer = ctx.enqueue_create_buffer[dtype](output_size)
    var exp_sum_buffer = ctx.enqueue_create_buffer[accum_type](heads)
    var qk_max_buffer = ctx.enqueue_create_buffer[accum_type](heads)
    ctx.enqueue_copy(q_buffer, q_host)
    ctx.enqueue_copy(k_buffer, k_host)
    ctx.enqueue_copy(v_buffer, v_host)
    ctx.enqueue_copy(out_buffer, out_host)

    var q = TileTensor(
        q_buffer.unsafe_ptr(), row_major[query_rows, depth]()
    ).as_imm()
    var k = TileTensorMHAOperand(
        TileTensor(
            k_buffer.unsafe_ptr(), row_major[1, keys, 1, depth]()
        ).as_imm()
    )
    var v = TileTensorMHAOperand(
        TileTensor(
            v_buffer.unsafe_ptr(), row_major[1, keys, 1, depth]()
        ).as_imm()
    )
    comptime kernel = _decode_kernel[dtype, type_of(k), type_of(v)]
    # Direct invocation covers this path even when dispatch selects FA4 on B200.
    ctx.enqueue_function[kernel](
        q.unsafe_ptr().as_unsafe_any_origin(),
        k,
        v,
        (out_buffer.unsafe_ptr() + guard).as_unsafe_any_origin(),
        exp_sum_buffer.unsafe_ptr().as_unsafe_any_origin(),
        qk_max_buffer.unsafe_ptr().as_unsafe_any_origin(),
        grid_dim=(1, 1, 1),
        block_dim=32,
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
    )
    ctx.enqueue_copy(out_host, out_buffer)
    ctx.synchronize()

    for head in range(heads):
        var max_score = Float64(-1e30)
        for key in range(keys):
            var dot = Float64(0)
            for col in range(depth):
                dot += Float64(q_host[head * depth + col]) * Float64(
                    k_host[key * depth + col]
                )
            scores[key] = dot * Float64(scale)
            max_score = max(max_score, scores[key])
        var denominator = Float64(0)
        for key in range(keys):
            scores[key] = exp(scores[key] - max_score)
            denominator += scores[key]
        for col in range(depth):
            var expected = Float64(0)
            for key in range(keys):
                expected += scores[key] * Float64(v_host[key * depth + col])
            assert_almost_equal(
                Float64(out_host[guard + head * depth + col]),
                expected / denominator,
                atol=2e-3,
                rtol=1e-2,
            )
    for i in range(guard):
        assert_equal(out_host[i], Scalar[dtype](42))
    for i in range(guard + heads * depth, output_size):
        assert_equal(out_host[i], Scalar[dtype](42))


def main() raises:
    with DeviceContext() as ctx:
        _check_decode[.bfloat16](ctx)
        _check_decode[.float16](ctx)
