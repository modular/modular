# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from math import ceildiv, isqrt
from sys import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from memory import UnsafePointer
from nn.normalization import *
from testing import assert_almost_equal, assert_true

from gpu.host._compile import _get_gpu_target

from utils.index import Index, IndexList


def compute_group_stats[
    t: DType
](vec: NDBuffer[t, 1], size: Int, eps: Scalar[t]) -> (Scalar[t], Scalar[t]):
    var sum_val = Scalar[t]()
    var sum_sq = Scalar[t]()
    for i in range(size):
        sum_val += vec[i]
        sum_sq += vec[i] * vec[i]
    var mean = sum_val / size
    var variance = max((sum_sq / size) - (mean * mean), 0.0)
    return (mean, isqrt(variance + eps))


fn run_group_norm_gpu[
    type: DType, rank: Int
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    num_groups: Int,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-5,
) raises:
    print("== run_group_norm_gpu")

    var N = shape[0]
    var C = shape[1]
    var spatial = shape.flattened_length() // (N * C)
    var group_size = C // num_groups * spatial
    var rows = N * num_groups
    var cols = group_size

    var data_h = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[type]].alloc(C)
    var beta_h = UnsafePointer[Scalar[type]].alloc(C)

    for i in range(rows * cols):
        data_h[i] = Scalar[type](i % 256)  # bounded range to avoid overflow

    for i in range(C):
        gamma_h[i] = ((i + C) / C).cast[type]()
        beta_h[i] = (i / C).cast[type]()

    var data_d = ctx.enqueue_create_buffer[type](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[type](C)
    var beta_d = ctx.enqueue_create_buffer[type](C)

    var param_shape = Index(C)
    var data_buf = NDBuffer[type, rank](data_d.unsafe_ptr(), shape)
    var gamma = NDBuffer[type, 1](gamma_d.unsafe_ptr(), param_shape)
    var beta = NDBuffer[type, 1](beta_d.unsafe_ptr(), param_shape)
    var epsilon = Scalar[type](1e-5)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, width]:
        return data_buf.load[width=width](rebind[IndexList[rank]](idx))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_scalar_fn[width: Int](idx: IndexList[1]) -> SIMD[type, width]:
        var val = gamma.load(Index(idx[0]))
        return SIMD[type, width](val)

    @__copy_capture(beta)
    @always_inline
    @parameter
    fn beta_scalar_fn[width: Int](idx: IndexList[1]) -> SIMD[type, width]:
        var val = beta.load(Index(idx[0]))
        return SIMD[type, width](val)

    group_norm[type, rank, input_fn, gamma_scalar_fn, beta_scalar_fn, "gpu"](
        shape, epsilon, num_groups, data_buf, ctx=ctx
    )
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = NDBuffer[type, 1](data_h + r * cols, cols)
        var stats = compute_group_stats(vec, cols, epsilon)
        var mean_ref = stats[0]
        var norm_factor = stats[1]
        for c in range(cols):
            var g = r % num_groups
            var c_base = g * (C // num_groups)
            var offset = c // spatial
            var true_c = c_base + offset
            var idx = r * cols + c
            var val = (
                (vec[c] - mean_ref) * norm_factor * gamma_h[true_c]
            ) + beta_h[true_c]
            assert_almost_equal(val, res[idx], rtol=rtol, atol=atol)

    _ = data_d^
    _ = gamma_d^
    _ = beta_d^
    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


def main():
    with DeviceContext() as ctx:
        alias default_simd = simdwidthof[
            DType.float32, target = _get_gpu_target()
        ]()

        # === Warp-Tiling Kernel Dispatch (SIMD-aligned, fits warp strategy) ===

        # Small, SIMD-aligned groups
        run_group_norm_gpu[DType.float32](ctx, Index(2, 8, 2, 2), num_groups=4)

        # Larger, but still small enough for warp tiling
        run_group_norm_gpu[DType.float32](ctx, Index(2, 32, 2, 2), num_groups=8)

        # SIMD aligned with group boundary, but not aligned with channel boundary
        run_group_norm_gpu[DType.float32](
            ctx, Index(2, 32, 1, 10), num_groups=8
        )

        # === Block Kernel Dispatch (too wide for warp or not divisible by SIMD width) ===

        # Large column count (too wide for warp)
        run_group_norm_gpu[DType.float32](
            ctx, Index(1, 128, 1, 64), num_groups=8
        )

        # Aligned, but still too large for warp strategy
        run_group_norm_gpu[DType.float32](
            ctx, Index(1, 64, 1, 64), num_groups=8
        )

        # Non-multiple of simd_width → scalar fallback block path
        run_group_norm_gpu[DType.float32](ctx, Index(2, 33, 1, 1), num_groups=1)

        # === Invalid Case: cols < simd_width → triggers safety assertion ===

        # Misaligned shape
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(1, 33, 1, 1), num_groups=11
            )
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )

        # Small group sizes result in too few columns
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(1, 12, 1, 1), num_groups=6
            )
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )
