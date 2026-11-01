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
"""SM100 blockwise FP8 matmul with 64-wide N scales (B200).

With `n_scale_granularity = 64` a 256-wide MMA tile spans four B-scale blocks
and a 192-wide tile spans three. The accumulator must apply every block to the
columns it covers, not just the first two. Each case compares the SM100 kernel
against the naive blockwise reference, which applies the scales per element,
with B-scales that differ between neighboring 64-wide blocks.

The dispatcher no longer picks a tile wider than twice the scale width, so the
wide tiles are forced through `blockwise_fp8_matmul` with an explicit config.
The cases through `matmul_dynamic_scaled_fp8` cover the dispatcher's own
choices. M is a multiple of 4 (the A-scales TMA needs a 16-byte row stride);
N = 5184 leaves a partial 256-wide tile at the end of the matrix.

The 192-wide tile over 128-wide scales starts every other tile 64 columns into
a scale block, which no dispatcher config does.
"""

from std.random import seed

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout._fillers import random
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    naive_blockwise_scaled_fp8_matmul,
)
from linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    GEMMKind,
    MatmulConfig,
)
from std.testing import assert_almost_equal

comptime K_G = 128


def _run_one[
    M: Int, N: Int, K: Int, N_G: Int = 64, mma_n: Int = 0
](ctx: DeviceContext) raises:
    """Runs one case; `mma_n == 0` goes through the dispatcher's tile choice."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scale_type = DType.float32

    print(
        "== test_matmul_sm100_blockwise_fp8_nscale64 (M=",
        M,
        ", N=",
        N,
        ", K=",
        K,
        ", N_G=",
        N_G,
        ", mma_n=",
        mma_n,
        ")",
        sep="",
    )

    comptime n_blocks = (N + N_G - 1) // N_G
    comptime k_blocks = K // K_G

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scales_size = k_blocks * M
    var b_scales_size = n_blocks * k_blocks

    var a_host = List(length=a_size, fill=Scalar[a_type](0))
    var b_host = List(length=b_size, fill=Scalar[b_type](0))
    var c_host = List(length=c_size, fill=Scalar[c_type](0))
    var c_host_ref = List(length=c_size, fill=Scalar[c_type](0))
    var a_scales_host = List(length=a_scales_size, fill=Scalar[scale_type](0))
    var b_scales_host = List(length=b_scales_size, fill=Scalar[scale_type](0))

    var a_host_tt = TileTensor(a_host, row_major[M, K]())
    var b_host_tt = TileTensor(b_host, row_major[N, K]())
    var a_scales_host_tt = TileTensor(a_scales_host, row_major[k_blocks, M]())
    var b_scales_host_tt = TileTensor(
        b_scales_host, row_major[n_blocks, k_blocks]()
    )
    random(a_host_tt)
    random(b_host_tt)
    random(a_scales_host_tt)
    random(b_scales_host_tt)
    # Spread the B-scales over several octaves so that a block scaled by one of
    # its neighbors' values shows up far above the comparison tolerance.
    for nb in range(n_blocks):
        for kb in range(k_blocks):
            var shift = Scalar[scale_type](1 << ((nb * 7 + kb * 3) % 5))
            b_scales_host[nb * k_blocks + kb] = (
                b_scales_host[nb * k_blocks + kb] * shift
            )

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scales_dev = ctx.enqueue_create_buffer[scale_type](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[scale_type](b_scales_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    var a_tt = TileTensor(a_dev, row_major[M, K]())
    var b_tt = TileTensor(b_dev, row_major[N, K]())
    var c_tt = TileTensor(c_dev, row_major[M, N]())
    var c_ref_tt = TileTensor(c_ref_dev, row_major[M, N]())
    var a_scales_tt = TileTensor(a_scales_dev, row_major[k_blocks, M]())
    var b_scales_tt = TileTensor(b_scales_dev, row_major[n_blocks, k_blocks]())

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk=Coord[Int, Int, Int](1, N_G, K_G),
    ](c_ref_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)
    ctx.synchronize()

    comptime if mma_n == 0:
        matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=1,
            n_scale_granularity=N_G,
            k_scale_granularity=K_G,
            transpose_b=True,
            target="gpu",
        ](c_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)
    else:
        comptime config = MatmulConfig[a_type, b_type, c_type, True](
            cluster_shape=(2, 1, 1),
            mma_shape=(128, mma_n, 32),
            cta_group=2,
            gemm_kind=GEMMKind.BLOCK_SCALED_1D2D_FP8,
        )
        blockwise_fp8_matmul[
            transpose_b=True,
            a_scales_type=scale_type,
            b_scales_type=scale_type,
            config=config,
            n_scale_granularity=N_G,
        ](c_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)
    ctx.synchronize()

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    var ndiff = 0
    for mi in range(M):
        for ni in range(N):
            var got = c_host[mi * N + ni]
            var want = c_host_ref[mi * N + ni]
            try:
                assert_almost_equal(got, want, rtol=1e-2, atol=1e-2)
            except:
                ndiff += 1
                if ndiff <= 3:
                    print(
                        "  diff @ (",
                        mi,
                        ",",
                        ni,
                        ") got=",
                        got,
                        " want=",
                        want,
                        sep="",
                    )
    if ndiff > 0:
        raise Error(
            String(
                "SM100 blockwise FP8 with ",
                N_G,
                "-wide N scales differs from the naive reference at ",
                ndiff,
                " of ",
                M * N,
                " positions (M=",
                M,
                ", N=",
                N,
                ")",
            )
        )
    print("  PASSED")


def _check[
    M: Int, N: Int, K: Int, N_G: Int = 64, mma_n: Int = 0
](ctx: DeviceContext, mut failures: Int) raises:
    try:
        _run_one[M, N, K, N_G, mma_n](ctx)
    except e:
        print("  FAILED:", e)
        failures += 1


def main() raises:
    seed(2758)
    var failures = 0
    with DeviceContext() as ctx:
        # 256-wide tile over 64-wide scales spans four blocks.
        _check[4, 5120, 1024, 64, 256](ctx, failures)
        _check[8, 5120, 1024, 64, 256](ctx, failures)
        _check[64, 7168, 1024, 64, 256](ctx, failures)
        _check[128, 7168, 1024, 64, 256](ctx, failures)
        # Partial last tile.
        _check[8, 5184, 1024, 64, 256](ctx, failures)
        # 192-wide tile over 64-wide scales spans three blocks.
        _check[8, 5120, 1024, 64, 192](ctx, failures)
        _check[64, 7168, 1024, 64, 192](ctx, failures)
        # Two blocks per tile: the pre-fix accumulator handled this.
        _check[64, 5120, 1024, 64, 128](ctx, failures)
        # The dispatcher's own choices for 64-wide scales.
        _check[4, 5120, 1024](ctx, failures)
        _check[64, 7168, 1024](ctx, failures)
        _check[128, 7168, 1024](ctx, failures)
        _check[8, 5184, 1024](ctx, failures)
        # Mid-M and large-M regimes use a 128-wide tile.
        _check[200, 5120, 1024](ctx, failures)
        _check[640, 5120, 1024](ctx, failures)
        # Narrow N keeps the 128-wide tile.
        _check[8, 1024, 1024](ctx, failures)
        # Tile starts 64 columns into a 128-wide scale block on odd tiles.
        _check[8, 5120, 1024, 128, 192](ctx, failures)
    if failures != 0:
        raise Error(String(failures, " case(s) failed"))
