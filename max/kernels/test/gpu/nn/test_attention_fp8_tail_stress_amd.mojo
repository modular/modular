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
"""Checks that AMD fp8 MHA and MLA attention keep small softmax weights.

Every 32nd key scores 64 * scale and the other keys score 0, so each of the
other keys has 2^-12 of a peak key's softmax weight. V is 0 for the peak keys
and 1 for the others, so every output element equals the share of the softmax
mass held by the non-peak keys. The kernels convert P to e4m3 before the P@V
MMA, and e4m3 rounds values below 2^-10 to zero, so these weights survive only
if P is scaled up before the conversion. Covers MHA decode and prefill, and MLA
decode.
"""

from std.math import exp, log
from std.collections import Optional
from std.testing import assert_true

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)

comptime FP8 = DType.float8_e4m3fn
comptime NUM_KEYS = 2048
comptime PEAK_STRIDE = 32
comptime SMALL_WEIGHT_LOG2 = 12


def _scale() -> Float32:
    return Float32(Float64(SMALL_WEIGHT_LOG2) * log(Float64(2.0)) / 64.0)


def _exact_share() -> Float64:
    var w = exp(-64.0 * Float64(_scale()))
    var peaks = NUM_KEYS // PEAK_STRIDE
    var small_mass = Float64(NUM_KEYS - peaks) * w
    return small_mass / (Float64(peaks) + small_mass)


def _check(name: String, share: Float64) raises:
    print(name, "share=", share)
    assert_true(abs(share - 1.0) < 0.01, String(name, " share=", share))


def mha_share[
    depth: Int, num_heads: Int
](ctx: DeviceContext, seq_len: Int) raises -> Float64:
    """Returns the MHA output mean divided by the exact small-weight share."""
    var q_size = seq_len * num_heads * depth
    var kv_size = NUM_KEYS * depth
    var q_h = ctx.enqueue_create_host_buffer[FP8](q_size)
    var k_h = ctx.enqueue_create_host_buffer[FP8](kv_size)
    var v_h = ctx.enqueue_create_host_buffer[FP8](kv_size)
    ctx.synchronize()
    for i in range(q_size):
        q_h[i] = Scalar[FP8](8) if i % depth == 0 else Scalar[FP8](0)
    for j in range(NUM_KEYS):
        var peak = j % PEAK_STRIDE == 0
        for d in range(depth):
            k_h[j * depth + d] = Scalar[FP8](8) if peak and d == 0 else Scalar[
                FP8
            ](0)
            v_h[j * depth + d] = Scalar[FP8](0) if peak else Scalar[FP8](1)

    var q_d = ctx.enqueue_create_buffer[FP8](q_size)
    var k_d = ctx.enqueue_create_buffer[FP8](kv_size)
    var v_d = ctx.enqueue_create_buffer[FP8](kv_size)
    var o_d = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(k_d, k_h)
    ctx.enqueue_copy(v_d, v_h)
    flash_attention(
        TileTensor(
            o_d.unsafe_ptr(),
            row_major((1, seq_len, Idx[num_heads], Idx[depth])),
        ),
        TileTensor(
            q_d.unsafe_ptr(),
            row_major((1, seq_len, Idx[num_heads], Idx[depth])),
        ),
        TileTensor(
            k_d.unsafe_ptr(), row_major((1, NUM_KEYS, Idx[1], Idx[depth]))
        ),
        TileTensor(
            v_d.unsafe_ptr(), row_major((1, NUM_KEYS, Idx[1], Idx[depth]))
        ),
        NullMask(),
        _scale(),
        ctx,
    )
    var o_h = ctx.enqueue_create_host_buffer[DType.bfloat16](q_size)
    ctx.enqueue_copy(o_h, o_d)
    ctx.synchronize()
    var acc: Float64 = 0
    for i in range(q_size):
        acc += o_h[i].cast[.float64]()
    _ = q_d
    _ = k_d
    _ = v_d
    return acc / Float64(q_size) / _exact_share()


def mla_share[
    num_heads: Int
](ctx: DeviceContext, num_partitions: Optional[Int]) raises -> Float64:
    """Returns the MLA decode output mean divided by the exact share."""
    comptime depth = 576
    comptime depth_v = 512
    var q_size = num_heads * depth
    var k_size = NUM_KEYS * depth
    var o_size = num_heads * depth_v
    var q_h = ctx.enqueue_create_host_buffer[FP8](q_size)
    var k_h = ctx.enqueue_create_host_buffer[FP8](k_size)
    ctx.synchronize()
    # Scores come from the first rope dimension; V is the latent part of K.
    for i in range(q_size):
        q_h[i] = Scalar[FP8](8) if i % depth == depth_v else Scalar[FP8](0)
    for j in range(NUM_KEYS):
        var peak = j % PEAK_STRIDE == 0
        for d in range(depth):
            var v: Scalar[FP8] = 0
            if d < depth_v:
                v = Scalar[FP8](0) if peak else Scalar[FP8](1)
            elif d == depth_v and peak:
                v = Scalar[FP8](8)
            k_h[j * depth + d] = v

    var q_d = ctx.enqueue_create_buffer[FP8](q_size)
    var k_d = ctx.enqueue_create_buffer[FP8](k_size)
    var o_d = ctx.enqueue_create_buffer[DType.bfloat16](o_size)
    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(k_d, k_h)
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads, _is_cache_length_accurate=True
    ](1, NUM_KEYS, 1, ctx)
    flare_mla_decoding[config=MHAConfig[FP8](num_heads, depth)](
        TileTensor(
            o_d.unsafe_ptr(), row_major((1, 1, Idx[num_heads], Idx[depth_v]))
        ).as_unsafe_any_origin(),
        TileTensor(
            q_d.unsafe_ptr(), row_major((1, 1, Idx[num_heads], Idx[depth]))
        ),
        TileTensor(
            k_d.unsafe_ptr(), row_major((1, NUM_KEYS, Idx[1], Idx[depth]))
        ),
        CausalMask(),
        _scale(),
        ctx,
        mla_args.gpu_tile_tensor(),
        num_partitions=num_partitions,
    )
    var o_h = ctx.enqueue_create_host_buffer[DType.bfloat16](o_size)
    ctx.enqueue_copy(o_h, o_d)
    ctx.synchronize()
    var acc: Float64 = 0
    for i in range(o_size):
        acc += o_h[i].cast[.float64]()
    _ = mla_args
    _ = q_d
    _ = k_d
    return acc / Float64(o_size) / _exact_share()


def main() raises:
    with DeviceContext() as ctx:
        _check("mha decode", mha_share[128, 16](ctx, 1))
        _check("mha prefill", mha_share[128, 16](ctx, 128))
        _check("mla decode np=1", mla_share[16](ctx, Optional[Int](1)))
        _check("mla decode np=4", mla_share[16](ctx, Optional[Int](4)))
        _check("mla decode default", mla_share[16](ctx, None))
