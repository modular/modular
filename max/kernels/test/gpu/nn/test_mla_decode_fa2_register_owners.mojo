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

"""Checks the generic MLA decode kernel against a host attention reference."""

from std.math import ceildiv, exp
from std.sys import size_of
from std.testing import assert_almost_equal, assert_equal
from std.utils.numerics import get_accum_type

from max.gpu.host import DeviceContext, FuncAttribute
from layout import TileTensor, row_major
from nn.attention.gpu.mla import mla_decoding_single_batch
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import TileTensorMHAOperand, MHAOperand

comptime BM = 16
comptime BN = 64
comptime BK = 64
comptime WM = 16
comptime WN = 16
comptime depth = 576
comptime depth_v = 512
comptime num_threads = (BM // WM) * (BN // WN) * 32


def _mla_decode_kernel[
    dtype: DType, k_t: MHAOperand, num_keys: Int
](
    q: MutPointer[Scalar[dtype], MutAnyOrigin],
    k: k_t,
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    exp_sum: MutPointer[Scalar[get_accum_type[dtype]()], MutAnyOrigin],
    qk_max: MutPointer[Scalar[get_accum_type[dtype]()], MutAnyOrigin],
):
    # Keep platform-width Int values inside the GPU launch boundary.
    mla_decoding_single_batch[
        BM=BM,
        BN=BN,
        BK=BK,
        WM=WM,
        WN=WN,
        depth=depth,
        depth_v=depth_v,
        num_threads=num_threads,
        num_pipeline_stages=6,
    ](q, k, output, exp_sum, qk_max, Float32(0.125), num_keys, 1, NullMask(), 0)


def _check_mla_decode[dtype: DType, keys: Int](ctx: DeviceContext) raises:
    comptime key_rows = ceildiv(keys, BN) * BN
    comptime q_size = BM * depth
    comptime k_size = key_rows * depth
    comptime out_size = BM * depth_v
    comptime guard = 32
    comptime scale = Float32(0.125)
    comptime accum_type = get_accum_type[dtype]()
    # Q, K and P tiles plus the cross-warp softmax scratch.
    comptime num_warps = num_threads // 32
    comptime smem = (BM * depth + BN * depth + BM * BN) * size_of[
        Scalar[dtype]
    ]() + 2 * num_warps * BM * size_of[Scalar[accum_type]]()

    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var k_host = ctx.enqueue_create_host_buffer[dtype](k_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](out_size + 2 * guard)
    var scores = ctx.enqueue_create_host_buffer[.float64](keys)

    for row in range(BM):
        for col in range(depth):
            q_host[row * depth + col] = Scalar[dtype](
                Float32((row * 3 + col) % 17 - 8) / 16
            )
    for row in range(key_rows):
        for col in range(depth):
            k_host[row * depth + col] = Scalar[dtype](
                Float32((row * 11 + col * 3) % 19 - 9) / 16
            ) if row < keys else Scalar[dtype](0)
    for i in range(out_size + 2 * guard):
        out_host[i] = Scalar[dtype](42)

    var q_buffer = ctx.enqueue_create_buffer[dtype](q_size)
    var k_buffer = ctx.enqueue_create_buffer[dtype](k_size)
    var out_buffer = ctx.enqueue_create_buffer[dtype](out_size + 2 * guard)
    var exp_sum_buffer = ctx.enqueue_create_buffer[accum_type](BM)
    var qk_max_buffer = ctx.enqueue_create_buffer[accum_type](BM)
    ctx.enqueue_copy(q_buffer, q_host)
    ctx.enqueue_copy(k_buffer, k_host)
    ctx.enqueue_copy(out_buffer, out_host)

    var k = TileTensorMHAOperand(
        TileTensor(
            k_buffer.unsafe_ptr(), row_major[1, key_rows, 1, depth]()
        ).as_imm()
    )
    comptime kernel = _mla_decode_kernel[dtype, type_of(k), keys]
    ctx.enqueue_function[kernel](
        q_buffer.unsafe_ptr().as_unsafe_any_origin(),
        k,
        out_buffer.unsafe_ptr().unsafe_offset(guard).as_unsafe_any_origin(),
        exp_sum_buffer.unsafe_ptr().as_unsafe_any_origin(),
        qk_max_buffer.unsafe_ptr().as_unsafe_any_origin(),
        grid_dim=(1, 1, 1),
        block_dim=num_threads,
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
    )
    ctx.enqueue_copy(out_host, out_buffer)
    ctx.synchronize()

    for head in range(BM):
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
        for col in range(depth_v):
            var expected = Float64(0)
            for key in range(keys):
                expected += scores[key] * Float64(k_host[key * depth + col])
            assert_almost_equal(
                Float64(out_host[guard + head * depth_v + col]),
                expected / denominator,
                atol=3e-3,
                rtol=2e-2,
            )
    for i in range(guard):
        assert_equal(out_host[i], Scalar[dtype](42))
    for i in range(guard + out_size, out_size + 2 * guard):
        assert_equal(out_host[i], Scalar[dtype](42))


def main() raises:
    with DeviceContext() as ctx:
        _check_mla_decode[.bfloat16, 128](ctx)
        _check_mla_decode[.bfloat16, 100](ctx)
